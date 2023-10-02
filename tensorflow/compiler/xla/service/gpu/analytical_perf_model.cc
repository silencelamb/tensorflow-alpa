#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <math.h>
#include "tensorflow/compiler/xla/service/gpu/analytical_perf_model.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

namespace xla {
namespace analytical_perf {

namespace py = pybind11;

std::string ToString(const PrimitiveType& type) {
  return primitive_util::LowercasePrimitiveTypeName(type);
}


// Expand the special replica_groups {{0}} to {{0,1,2,..,n}}
const std::vector<ReplicaGroup> ExpandSpecialReplicaGroups(
    const std::vector<ReplicaGroup>& replica_groups, int64_t num_devices) {
  if (replica_groups.size() == 1 && replica_groups[0].replica_ids_size() == 1 &&
      num_devices != 1) {
    ReplicaGroup group;
    for (int64_t i = 0; i < num_devices; ++i) {
      group.add_replica_ids(i);
    }
    return {group};
  } else {
    return replica_groups;
  }
}

// Make a string key of a replica_groups.
std::string Group2Str(const std::vector<ReplicaGroup>& replica_groups) {
  std::ostringstream os;

  os << "(";
  for (const auto& group : replica_groups) {
    os << "(";
    for (const auto& id : group.replica_ids()) {
      os << id << ",";
    }
    os << "),";
  }
  os << ")";

  return os.str();
}



double AnalyticalPerfOfHloModule(const HloModule* hlo_module) {
  const int64_t num_devices = hlo_module->config().num_partitions();
  int verbose = pass_context::GetInt("analytical_perf::verbose", 0);
  std::string hardware = pass_context::GetString("analytical_perf::hardware", "gpu");

  // parse python dict to cpp dict
  py::dict compute_py_dict = pass_context::GetPyObject("analytical_perf::compute_dict");
  std::map<PrimitiveType, int64_t> compute_dict;
  PyGILState_STATE gstate = PyGILState_Ensure();
  for (auto item : compute_py_dict) {
    PrimitiveType p_type = PrimitiveType(py::cast<int64_t>(item.first));
    int64_t compute = py::cast<int64_t>(item.second);
    compute_dict[p_type] = compute;
  }
  PyGILState_Release(gstate);

  int num_micro_batches =
      pass_context::GetInt("analytical_perf::num_micro_batches", 1);
  std::string grad_sync_channel_ids =
      pass_context::GetString("analytical_perf::grad_sync_channel_ids", "");
  bool force_use_fp16 = pass_context::GetBool("analytical_perf::force_use_fp16", false);

  // hardware == "gpu"
  int64_t card_num = pass_context::GetInt("analytical_perf_gpu::card_num", 8);
  int64_t card_bw= pass_context::GetInt("analytical_perf_gpu::card_bw", pow(10, 9));
  int64_t card_mem = pass_context::GetInt("analytical_perf_gpu::card_mem", pow(10, 9));
  int64_t node_bw = pass_context::GetInt("analytical_perf_gpu::node_bw", pow(10, 9));
  // hardware == "wsc"
  int64_t tile_r_num = pass_context::GetInt("analytical_perf_wsc::tile_r_num", 6);
  int64_t tile_c_num = pass_context::GetInt("analytical_perf_wsc::tile_c_num", 6);
  int64_t tile_bw= pass_context::GetInt("analytical_perf_wsc::tile_bw", pow(10, 9));
  int64_t tile_mem = pass_context::GetInt("analytical_perf_wsc::tile_mem", pow(10, 9));
  int64_t die_bw = pass_context::GetInt("analytical_perf_wsc::die_bw", pow(10, 9));
  // common
  double cmp_ul = pass_context::GetDouble("analytical_perf::cmp_ul");
  double bw_ul = pass_context::GetDouble("analytical_perf::bw_ul");
  gpu::Node gpu_node(card_num, compute_dict, card_bw, card_mem, node_bw, cmp_ul, bw_ul);
  wsc::Die wsc_die(tile_r_num, tile_c_num, compute_dict, tile_bw, tile_mem, die_bw, cmp_ul, bw_ul);


  // Compute cost of all instruction.
  double sum = 0.0;
  const HloComputation* entry_computation = hlo_module->entry_computation();
  for (const HloInstruction* ins : entry_computation->instructions()) {
    double cost = 0.0;

    if (ins->opcode() == HloOpcode::kAllGather ||
        ins->opcode() == HloOpcode::kAllReduce ||
        ins->opcode() == HloOpcode::kAllToAll ||
        ins->opcode() == HloOpcode::kReduceScatter) {
      auto coll = DynCast<HloCollectiveInstruction>(ins);
      CHECK(coll != nullptr);

      std::vector<ReplicaGroup> replica_groups = coll->replica_groups();
      // Expand the special replica_groups {{0}}
      replica_groups = ExpandSpecialReplicaGroups(replica_groups, num_devices);
      std::string group_str = Group2Str(replica_groups);
      if (replica_groups.size() > 1) {
        std::cout << "GroupString(unhandled cases): " << group_str << ": ";
      }
      if (verbose) {
        if (replica_groups.size() <= 1) {
          std::cout << "GroupString: " <<  group_str << ": ";
        }
      }
      for (const auto operand : ins->operands()) {
        int64_t size = spmd::GetBytes(operand->shape());
        double tmp_op_time;
        double normalizer = 1.0;
        COMM_MODE comm_mode = ALL_REDUCE;
        switch (ins->opcode()) {
          case HloOpcode::kAllGather:
            comm_mode = ALL_GATHER;
            break;
          case HloOpcode::kAllReduce: {
            // Amortize the cost of grad sync with the number of micro batches.
            std::string key = absl::StrFormat(".%d.", *ins->channel_id());
            if (grad_sync_channel_ids.find(key) != std::string::npos) {
              normalizer = num_micro_batches;
            }
            comm_mode = ALL_REDUCE;
            break;
          }
          case HloOpcode::kAllToAll:
            comm_mode = ALL_TO_ALL;
            break;
          case HloOpcode::kReduceScatter:
            comm_mode = REDUCE_SCATTER;
            break;
          default:
            break;
        }
        if (verbose == 2) {
          std::cout << "comm:" << ins->opcode() <<  "-" << size << ":device-"
                               << num_devices << ":";
        }
        if (hardware == "gpu") {
          int64_t node_num = int(floor((num_devices / gpu_node.cards.size())));
          if (verbose == 2) {
             std::cout << "node_num-" << node_num << ":";
          }
          if (node_num > 1) {
            tmp_op_time = gpu_node.AnalyseCommunicateTime(size, comm_mode, node_num, verbose);
            tmp_op_time /= normalizer;
          }
          else {
            tmp_op_time = gpu_node.cards[0].AnalyseCommunicateTime(size, comm_mode, num_devices);
            std::cout << "tmp_op_time-" << tmp_op_time << ":normalizer-" << normalizer << ":";
            tmp_op_time /= normalizer;
          }
        }
        else if (hardware == "wsc") {
          tmp_op_time = wsc_die.AnalyseCommunicateTime(size, comm_mode, num_devices);
          tmp_op_time /= normalizer;     
        }
        cost += tmp_op_time;
      }
    }

    if (ins->IsCustomCall(xla::gpu::kGemmCallTarget) || ins->opcode() == HloOpcode::kDot) {
      const HloInstruction* lhs = ins->operand(0);
      const HloInstruction* rhs = ins->operand(1);
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      xla::DotDimensionNumbers dnum;
      if (ins->IsCustomCall(xla::gpu::kGemmCallTarget)){
        auto config = ins->backend_config<xla::gpu::GemmBackendConfig>().ValueOrDie();
        dnum = config.dot_dimension_numbers();
      }
      if (ins->opcode() == HloOpcode::kDot) {
        dnum = ins->dot_dimension_numbers();
      }

      std::tie(lhs_space_dims, rhs_space_dims) =
          spmd::GetSpaceDims(lhs->shape(), rhs->shape(), dnum);

      int64_t flop_count =
          lhs->shape().dimensions(lhs_space_dims[0]) *
          lhs->shape().dimensions(dnum.lhs_contracting_dimensions(0)) *
          rhs->shape().dimensions(rhs_space_dims[0]);
      for (int64_t dim : dnum.lhs_batch_dimensions()) {
        flop_count *= lhs->shape().dimensions(dim);
      }
      flop_count *= 2;

      if (hardware == "gpu") {
        cost += gpu_node.cards[0].AnalyseComputeTime(flop_count, ins->shape().element_type(), 1, force_use_fp16);
      }
      else {
        cost += wsc_die.AnalyseComputeTime(flop_count, ins->shape().element_type(), 1, force_use_fp16);
      }
    }
    if (verbose == 2) {
      std::cout << ins->opcode() <<  " " <<std::fixed << std::setprecision(8) 
                                 << cost << std::endl;
    }
    if (cost > 0) {
      spmd::StdCerr(verbose) << ins->ToString() << " cost(in SPMD): " << std::fixed
                             << std::setprecision(8) << cost << std::endl;
    }

    sum += cost;
  }

  return sum;
}

}  // namespace gpu
}  // namespace xla
