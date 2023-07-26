#include <pybind11/stl.h>
#include "tensorflow/compiler/xla/service/gpu/gpu_cost_model.h"
#include "tensorflow/compiler/xla/service/gpu/analytical_perf_model.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"

namespace xla {
namespace gpu {

namespace py = pybind11;

std::string ToString(const PrimitiveType& type) {
  return primitive_util::LowercasePrimitiveTypeName(type);
}

// Make a string key of a replica_groups.
std::string Group2Str(py::object tuple) {
  if (tuple.is_none()) {
    return "()";
  }

  py::tuple replica_groups = py::cast<py::tuple>(tuple);
  std::ostringstream os;
  os << "(";
  for (const auto& group : replica_groups) {
    os << "(";
    for (const auto& id : py::cast<py::tuple>(group)) {
      os << py::cast<int64_t>(id) << ",";
    }
    os << "),";
  }
  os << ")";

  return os.str();
}

// Make a string key of a replica_groups.
std::string Group2Str(const std::vector<std::vector<int>>& replica_groups) {
  std::ostringstream os;

  os << "(";
  for (const auto& group : replica_groups) {
    os << "(";
    for (const auto& id : group) {
      os << id << ",";
    }
    os << "),";
  }
  os << ")";

  return os.str();
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

// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // Construct the class from the corresponding python object
  // alpa/mesh_profiling.py::ProfilingResult.
  ProfilingResult(py::object prof_result) {
    if (!prof_result.is_none()) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      {
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_gather_cost_dict")),
            all_gather_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_reduce_cost_dict")),
            all_reduce_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_to_all_cost_dict")),
            all_to_all_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("reduce_scatter_cost_dict")),
            reduce_scatter_cost_dict_);
        CommDictPyToCpp(py::cast<py::dict>(prof_result.attr("dot_cost_dict")),
                        dot_cost_dict_);
      }
      PyGILState_Release(gstate);
    }
  }

  bool Enabled() const { return enabled_; }

  double EstimateAllGatherCost(const std::vector<ReplicaGroup>& replica_groups,
                               int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(const std::vector<ReplicaGroup>& replica_groups,
                               int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateAllToAllCost(const std::vector<ReplicaGroup>& replica_groups,
                              int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_to_all_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_to_all_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<ReplicaGroup>& replica_groups, int64_t size,
      PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
  }

  double EstimateDotCost(int64_t flop_count, PrimitiveType dtype) {
    std::vector<ReplicaGroup> fake_replica_groups;
    return EstimateInternal(fake_replica_groups, flop_count, dtype,
                            dot_cost_dict_) -
           EstimateInternal(fake_replica_groups, 0, dtype, dot_cost_dict_);
  }

  std::string ToString() {
    std::ostringstream os;
    os << "all-reduce cost dict:\n";
    for (const auto& item : all_reduce_cost_dict_) {
      os << "key: (" << item.first.first << ", "
         << gpu::ToString(item.first.second) << ")\n";
    }
    os << "dot cost dict:\n";
    for (const auto& item : dot_cost_dict_) {
      os << "key: (" << item.first.first << ", "
         << gpu::ToString(item.first.second) << ")\n";
    }
    return os.str();
  }

 private:
  // Dictionary type for communicaiton cost.
  // Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
  // pair<group, dtype>
  using CommDictKey = std::pair<std::string, PrimitiveType>;
  // vector<pair<size, time>>
  using CommDictValue = std::vector<std::pair<int64_t, double>>;
  using CommDict = absl::flat_hash_map<CommDictKey, CommDictValue>;

  // Estimate the cost by linear interpolation bewteen the two closest points.
  double EstimateInternal(const std::vector<ReplicaGroup>& replica_groups,
                          int64_t size, PrimitiveType dtype,
                          const CommDict& cost_dict) const {
    if (dtype != PrimitiveType::F16 && dtype != PrimitiveType::F32) {
      // Cast other types to F32.
      dtype = PrimitiveType::F32;
    }

    CommDictKey key(Group2Str(replica_groups), dtype);
    if (!cost_dict.count(key)) {
      LOG(WARNING) << "Warning: cannot find key: (" << key.first << ", "
                   << gpu::ToString(key.second) << ")" << std::endl;
      return size;
    }
    CommDictValue cost_list = cost_dict.at(key);

    CHECK(!cost_list.empty());

    size_t i;
    if (size > cost_list.back().first) {
      i = cost_list.size() - 2;
    } else if (size < cost_list.front().first) {
      i = 0;
    } else {
      for (i = 0; i < cost_list.size() - 1; ++i) {
        if (cost_list[i].first <= size && size <= cost_list[i + 1].first) {
          break;
        }
      }
    }

    int64_t left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64_t right_size = cost_list[i + 1].first;
    double right_cost = cost_list[i + 1].second;

    return 1.0 * (size - left_size) / (right_size - left_size) *
               (right_cost - left_cost) +
           left_cost;
  }

  // Convert a python communication cost dict to c++ dict.
  void CommDictPyToCpp(py::dict py_dict, CommDict& cpp_dict) {
    // the type of py_dict: Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
    for (auto item : py_dict) {
      py::tuple tuple_key = py::cast<py::tuple>(item.first);
      std::string dtype_str = py::cast<std::string>(tuple_key[1]);
      PrimitiveType dtype;

      if (dtype_str == "f16") {
        dtype = PrimitiveType::F16;
      } else if (dtype_str == "f32") {
        dtype = PrimitiveType::F32;
      } else {
        LOG(FATAL) << "Invalid dtype: " << dtype_str;
      }

      CommDictKey key(Group2Str(tuple_key[0]), dtype);

      py::list list_val = py::cast<py::list>(item.second);
      for (const auto x : list_val) {
        py::tuple tuple_val = py::cast<py::tuple>(x);
        cpp_dict[key].push_back(std::make_pair(py::cast<int64_t>(tuple_val[0]),
                                               py::cast<double>(tuple_val[1])));
      }
    }
  }

  bool enabled_;
  CommDict all_reduce_cost_dict_;
  CommDict all_gather_cost_dict_;
  CommDict reduce_scatter_cost_dict_;
  CommDict all_to_all_cost_dict_;
  CommDict dot_cost_dict_;  // Reuse CommDict data structure for dot.
};

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

double EstimateHloModuleCost(const HloModule* hlo_module) {
  // Load profiling results.
  ProfilingResult prof_result(
      pass_context::GetPyObject("gpu_cost_model::profiling_results"));
  const int64_t num_devices = hlo_module->config().num_partitions();
  int verbose = pass_context::GetInt("gpu_cost_model::verbose", 0);
  int num_micro_batches =
      pass_context::GetInt("gpu_cost_model::num_micro_batches", 1);
  std::string grad_sync_channel_ids =
      pass_context::GetString("gpu_cost_model::grad_sync_channel_ids", "");

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

      for (const auto operand : ins->operands()) {
        int64_t size = spmd::GetBytes(operand->shape());
        switch (ins->opcode()) {
          case HloOpcode::kAllGather:
            cost += prof_result.EstimateAllGatherCost(
                replica_groups, size, operand->shape().element_type());
            break;
          case HloOpcode::kAllReduce: {
            double normalizer = 1.0;

            // Amortize the cost of grad sync with the number of micro batches.
            std::string key = absl::StrFormat(".%d.", *ins->channel_id());
            if (grad_sync_channel_ids.find(key) != std::string::npos) {
              normalizer = num_micro_batches;
            }

            cost += prof_result.EstimateAllReduceCost(
                        replica_groups, size, operand->shape().element_type()) /
                    normalizer;
            break;
          }
          case HloOpcode::kAllToAll:
            cost += prof_result.EstimateAllToAllCost(
                replica_groups, size, operand->shape().element_type());
            break;
          case HloOpcode::kReduceScatter:
            cost += prof_result.EstimateReduceScatterCost(
                replica_groups, size, operand->shape().element_type());
            break;
          default:
            break;
        }
      }
    }

    if (ins->IsCustomCall(kGemmCallTarget)) {
      const HloInstruction* lhs = ins->operand(0);
      const HloInstruction* rhs = ins->operand(1);
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      auto config = ins->backend_config<GemmBackendConfig>().ValueOrDie();
      auto dnum = config.dot_dimension_numbers();
      std::tie(lhs_space_dims, rhs_space_dims) =
          spmd::GetSpaceDims(lhs->shape(), rhs->shape(), dnum);

      int64_t flop_count =
          lhs->shape().dimensions(lhs_space_dims[0]) *
          lhs->shape().dimensions(dnum.lhs_contracting_dimensions(0)) *
          rhs->shape().dimensions(rhs_space_dims[0]);
      for (int dim : dnum.lhs_batch_dimensions()) {
        flop_count *= lhs->shape().dimensions(dim);
      }
      flop_count *= 2;

      cost +=
          prof_result.EstimateDotCost(flop_count, ins->shape().element_type());
    }

    if (cost > 0) {
      spmd::StdCerr(verbose) << ins->ToString() << " cost: " << std::fixed
                             << std::setprecision(8) << cost << std::endl;
    }

    sum += cost;
  }

  return sum;
}

// added by Zijun Xu
// this function is used for estimating the peak memory cost of a given HloModule 
StatusOr<double> EstimateHloModuleMemory(HloModule *module) {

  auto size_fn = [](const BufferValue& buffer) {
    return xla::spmd::GetBytes(buffer.shape());
  };
  
  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using DependencyHloOrdering).
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module, size_fn,
                                     ComputationSchedulerToModuleScheduler(
                                         DFSMemoryScheduler)));
                                         
  const HloComputation* entry_computation = module->entry_computation();

  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module).ConsumeValueOrDie();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, *alias_analysis, entry_computation));
  
  return hlo_live_range->ComputePeakMemoryCost();
}

py::dict convertHloCostToPydict(const HloCost & hlo_cost){
  py::dict result;
  result["op_name"] = py::cast(hlo_cost.op_name);
  result["flops"] = py::cast(hlo_cost.flops);
  py::dict network_describ;
  network_describ["comm_type"] = py::cast(hlo_cost.network_describ.comm_type);
  network_describ["replica_groups_str"] = py::cast(hlo_cost.network_describ.replica_groups_str);
  network_describ["comm_data_count"] = py::cast(hlo_cost.network_describ.comm_data_count);
  result["network_describ"] = network_describ;
  result["data_type"] = py::cast(hlo_cost.data_type);
  result["network_count"] = py::cast(hlo_cost.network_count);
  result["estimated_time"] = py::cast(hlo_cost.estimated_time);
  return result;
}


py::list HloModuleCost(const HloModule* hlo_module) {
  py::list hlo_cost_result;

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
      pass_context::GetInt("gpu_cost_model::num_micro_batches", 1);
  std::string grad_sync_channel_ids =
      pass_context::GetString("gpu_cost_model::grad_sync_channel_ids", "");
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
  xla::analytical_perf::gpu::Node gpu_node(card_num, compute_dict, card_bw, card_mem, node_bw, cmp_ul, bw_ul);
  xla::analytical_perf::wsc::Die wsc_die(tile_r_num, tile_c_num, compute_dict, tile_bw, tile_mem, die_bw, cmp_ul, bw_ul);


  // Compute cost of all instruction.
  double flops_sum = 0.0, mem_sum = 0.0, network_sum = 0.0;
  const HloComputation* entry_computation = hlo_module->entry_computation();
  for (const HloInstruction* ins : entry_computation->instructions()) {
    double cost = 0.0;
    HloCost tmp_hlo_cost;
    double tmp_op_time = 0.0;

    if (ins->opcode() == HloOpcode::kAllGather ||
        ins->opcode() == HloOpcode::kAllReduce ||
        ins->opcode() == HloOpcode::kAllToAll ||
        ins->opcode() == HloOpcode::kReduceScatter) {
      auto coll = DynCast<HloCollectiveInstruction>(ins);
      CHECK(coll != nullptr);

      std::vector<ReplicaGroup> replica_groups = coll->replica_groups();
      // Expand the special replica_groups {{0}}
      replica_groups = ExpandSpecialReplicaGroups(replica_groups, num_devices);
      std::string replica_groups_str = Group2Str(replica_groups);

      for (const auto operand : ins->operands()) {
        int64_t size = spmd::GetBytes(operand->shape());
        double normalizer = 1.0;
        xla::analytical_perf::COMM_MODE comm_mode = xla::analytical_perf::COMM_MODE::ALL_REDUCE;
        std::string comm_str;

        switch (ins->opcode()) {
          case HloOpcode::kAllGather:{
            comm_str = "AllGather";
            comm_mode = xla::analytical_perf::COMM_MODE::ALL_GATHER;
            break;
          }
          case HloOpcode::kAllReduce: {
            // Amortize the cost of grad sync with the number of micro batches.
            std::string key = absl::StrFormat(".%d.", *ins->channel_id());
            if (grad_sync_channel_ids.find(key) != std::string::npos) {
              normalizer = num_micro_batches;
            }
            comm_str = "AllReduce";
            comm_mode = xla::analytical_perf::COMM_MODE::ALL_REDUCE;
            break;
          }
          case HloOpcode::kAllToAll:
            comm_str = "AllToAll";
            comm_mode = xla::analytical_perf::COMM_MODE::ALL_TO_ALL;
            break;
          case HloOpcode::kReduceScatter:
            comm_str = "ReduceScatter";
            comm_mode = xla::analytical_perf::COMM_MODE::REDUCE_SCATTER;
            break;
          default:
            break;
        }
        // emtimate time
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
        double network_count = xla::analytical_perf::get_comm_total_size(size, comm_mode, num_devices);
        tmp_hlo_cost.set_value(ins->name(), 0.0, comm_str, replica_groups_str, size, operand->shape().element_type(), network_count, tmp_op_time); 
        py::dict py_hlo_cost = convertHloCostToPydict(tmp_hlo_cost);
        hlo_cost_result.append(py_hlo_cost);
      }
    }

    if (ins->IsCustomCall(kGemmCallTarget)) {
      const HloInstruction* lhs = ins->operand(0);
      const HloInstruction* rhs = ins->operand(1);
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      auto config = ins->backend_config<GemmBackendConfig>().ValueOrDie();
      auto dnum = config.dot_dimension_numbers();
      std::tie(lhs_space_dims, rhs_space_dims) =
          spmd::GetSpaceDims(lhs->shape(), rhs->shape(), dnum);

      int64_t flop_count =
          lhs->shape().dimensions(lhs_space_dims[0]) *
          lhs->shape().dimensions(dnum.lhs_contracting_dimensions(0)) *
          rhs->shape().dimensions(rhs_space_dims[0]);
      for (int dim : dnum.lhs_batch_dimensions()) {
        flop_count *= lhs->shape().dimensions(dim);
      }
      flop_count *= 2;
      if (hardware == "gpu") {
        tmp_op_time = gpu_node.cards[0].AnalyseComputeTime(flop_count, ins->shape().element_type(), 1, force_use_fp16);
      }
      else {
        tmp_op_time = wsc_die.AnalyseComputeTime(flop_count, ins->shape().element_type(), 1, force_use_fp16);
      }
      tmp_hlo_cost.set_flops_network(ins->name(), flop_count, ins->shape().element_type(), 0.0, tmp_op_time);
      flops_sum += flop_count;
      py::dict py_hlo_cost = convertHloCostToPydict(tmp_hlo_cost);
      hlo_cost_result.append(py_hlo_cost);
    }
  }
  HloCost tmp_hlo_cost;
  // tmp_hlo_cost.set_flops_network(ins->name(), flops_sum, ins->shape().element_type());
  tmp_hlo_cost.set_flops_network("total", flops_sum, 1, 0.0);
  py::dict py_hlo_cost = convertHloCostToPydict(tmp_hlo_cost);
  hlo_cost_result.append(py_hlo_cost);

  return hlo_cost_result;
}


}  // namespace gpu
}  // namespace xla
