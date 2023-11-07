#include <bits/stdint-intn.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <math.h>
#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include "tensorflow/compiler/xla/service/gpu/analytical_perf_model.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"

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
  double die_bw = (double) pass_context::GetInt("analytical_perf_wsc::die_bw", pow(10, 9));
  double die_alpha = pass_context::GetDouble("analytical_perf_wsc::die_alpha");
  double die_beta = 1.0 / die_bw;
  std::vector<int64_t> submesh_shape = pass_context::GetIntVector("analytical_perf_wsc::mesh_shape");
  // common
  double cmp_ul = pass_context::GetDouble("analytical_perf::cmp_ul");
  double bw_ul = pass_context::GetDouble("analytical_perf::bw_ul");
  gpu::Node gpu_node(card_num, compute_dict, card_bw, card_mem, node_bw, cmp_ul, bw_ul);
  wsc::Die wsc_die(tile_r_num, tile_c_num, compute_dict, tile_bw, tile_mem, die_bw, cmp_ul, bw_ul, die_alpha, die_beta);

  // whether use the greedy search collective cost
  bool use_greedy_collective_cost = pass_context::GetBool("analytical_perf::use_greedy_coll_cost", false);
  std::map<std::pair<int, std::pair<int, int>>, int> collective_cost_map;
  if (use_greedy_collective_cost) {
    py::dict collective_cost_dict = pass_context::GetPyObject("collective_cost_dict");
    
    PyGILState_STATE gstate2 = PyGILState_Ensure();
    for (auto item : collective_cost_dict) {
      py::tuple key = py::cast<py::tuple>(item.first);
      int coll = key[0].cast<int>();
      py::tuple key_1 = py::cast<py::tuple>(key[1]); 
      auto mesh_shape = std::make_pair(key_1[0].cast<int>(), key_1[1].cast<int>());
      collective_cost_map[std::make_pair(coll, mesh_shape)] = py::cast<int>(item.second);
    }
    PyGILState_Release(gstate2);
    // for (const auto &item : collective_cost_map) {
    //   int coll = item.first.first;
    //   int m = item.first.second.first;
    //   int n = item.first.second.second;
    //   std::cout<<coll<<":("<<m<<" "<<n<<") "<<item.second<<std::endl;
    // }
  }

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
      // if (replica_groups.size() > 1) {
      //   std::cout << "GroupString(unhandled cases): " << group_str << ": ";
      // }
      if (verbose) {
        if (replica_groups.size() <= 1) {
          std::cout << "GroupString: " <<  group_str << ": ";
        }
      }
      for (const auto operand : ins->operands()) {
        int64_t size = spmd::GetBytes(operand->shape());
        double tmp_op_time = 0;
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
          if (use_greedy_collective_cost == true) {
            std::pair<int, int> mesh_shape(0, 0);
            if (replica_groups.size() == 1) {
              mesh_shape = std::make_pair(submesh_shape[0], submesh_shape[1]);
            }
            else{
              mesh_shape = std::make_pair(1, replica_groups[0].replica_ids_size());
            }
            
            auto comm_cost_key = std::make_pair(comm_mode, mesh_shape);
            int time_steps = collective_cost_map[comm_cost_key];
            // std::cout<<size<<" "<<comm_mode<<" "<<num_devices<<std::endl;
            tmp_op_time = wsc_die.AnalyseCommunicateTimeGreedy(size, comm_mode, num_devices, time_steps);
          } else {
            tmp_op_time = wsc_die.AnalyseCommunicateTime(size, comm_mode, num_devices);
            tmp_op_time /= normalizer;     
          }
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

// ------ Parameter Analysis ------
std::pair<double, double> MemoryOffloader::ParameterAnalysis(const HloModule* module) const {
  double params_size = 0.0;
  double max_param_size = 0.0;
  const HloComputation* entry_comp = module->entry_computation();
  for (auto inst : entry_comp->parameter_instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      auto shape = inst->shape();
      if (force_use_fp16_) 
        shape.set_element_type(PrimitiveType::F16);
      if (shape.is_dynamic())
        continue;
      max_param_size = std::max(spmd::GetBytes(shape), max_param_size);
      params_size += spmd::GetBytes(shape);
    }
  }

  return {params_size, max_param_size};
}

// ------ Get the hlo module alloc memory ------
double MemoryOffloader::GetHloMoudleAllocMemory(const HloModule* module) {
  // Size function for buffer
  bool force_use_fp16 = force_use_fp16_;
  auto size_fn = [&force_use_fp16](const BufferValue& buffer) {
    auto new_shape = Shape(buffer.shape());
    if (force_use_fp16) {
      new_shape.set_element_type(PrimitiveType::F16);
    }
    return spmd::GetBytes(new_shape);
  };

  // create module scheduler and do buffer assign
  HloSchedule schedule = ScheduleModule(module, size_fn, 
                                        ComputationSchedulerToModuleScheduler(DFSMemoryScheduler)).value();
  auto buffer_assign = BufferAssigner::Run(module, 
                                          std::make_unique<SequentialHloOrdering>(schedule),
                                          size_fn,
                                          [](LogicalBuffer::Color) { return 1; },
                                           /*allocate_buffers_for_constants=*/true).value();
  double alloc_memory = buffer_assign->GetStats().total_allocation_bytes;
  return alloc_memory;
}

// ------ Helper func to slice hlo module ------
// This function can slice pass-in module to [start_idx, end_idx) part module.
// NOTE: index assumes to be PostOrder index obtained from module entry computation.
std::unique_ptr<HloModule> SliceHloModule(const HloModule* module, int64_t start_idx, int64_t end_idx) {
  const HloComputation* entry_comp = module->entry_computation();
  auto comp_instructions = entry_comp->MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*, int64_t> inst_2_index;

  // Index each instruction
  int64_t counter = 0;
  for (auto inst : comp_instructions) {
    inst_2_index.insert({inst, counter});
    counter++;
  }

  // Get sliced instructions & corresponding ret instrucutions
  std::vector<const HloInstruction*> sliced_instructions;
  std::vector<const HloInstruction*> sliced_ret_instructions;
  std::vector<const HloInstruction*> sliced_params;

  if (end_idx - start_idx == 1) {
    if (comp_instructions[start_idx]->opcode() == HloOpcode::kParameter) {
      spmd::StdCerr(2) << "Warning: Hit single param case: " << start_idx << "\n"; 
    }
  }

  for (auto inst : comp_instructions) {
    auto index = inst_2_index[inst];
    // Sliced module [start_idx, end_idx)
    if (index >= start_idx && index < end_idx) {
      // for internal inst with external input, should be param
      if (inst->opcode() != HloOpcode::kParameter) {
        for (auto operand : inst->operands()) {
          if (inst_2_index[operand] < start_idx) {
            sliced_instructions.push_back(operand);
            sliced_params.push_back(operand);
          }
        }
        sliced_instructions.push_back(inst);
      }
      // Set sliced module ret insts
      int64_t last_user_idx = -1;
      int64_t first_user_idx = INT64_MAX;
      for (auto user : inst->users()) {
        CHECK(inst_2_index.contains(user));
        last_user_idx = std::max(last_user_idx, inst_2_index[user]);
        first_user_idx = std::min(first_user_idx, inst_2_index[user]);
      }
      // if inst used in after end_idx, should be saved out.
      if (inst->opcode() != HloOpcode::kParameter && last_user_idx != -1 && last_user_idx >= end_idx) {
        sliced_ret_instructions.push_back(inst);
      }
      // If param first user before end_idx, should be sliced param.
      if (inst->opcode() == HloOpcode::kParameter && first_user_idx != INT64_MAX
          && first_user_idx < end_idx) {
        sliced_params.push_back(inst);
        sliced_instructions.push_back(inst);
      }
    } 
  }

  if (sliced_instructions.size() == 0) {
    spmd::StdCerr(2) << "Error: cause sliced instructions zero "  << "\n"; 
    return nullptr;
  }
  
  // Steup the new sliced module
  HloModuleConfig config = module->config();
  config.set_shardable_value_update_pairs({});
  config.mutable_fusion_config()->clear();
  config.mutable_dot_config()->clear();
  config.mutable_layout_config()->clear();

  const std::string sliced_suffix = absl::StrCat("sliced-", start_idx, "-", end_idx);
  auto sliced_module = absl::make_unique<HloModule>(absl::StrCat(module->name(), "-", 
                                                              sliced_suffix), config);
  auto context_ptr = absl::make_unique<HloCloneContext>(sliced_module.get(), sliced_suffix);
  HloCloneContext* context = context_ptr.get();
  std::vector<std::unique_ptr<HloInstruction>> instructions;
  
  // Create new params for new sliced module
  int n_parameters = sliced_params.size();
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> old_param_2_new;
  for (int i = 0; i < n_parameters; i++) {
    auto ori_param = sliced_params[i];
    auto new_param = HloInstruction::CreateParameter(i, ori_param->shape(), 
                                                      absl::StrCat("param_", i));
    if (ori_param->has_sharding()) {
      new_param->set_sharding(ori_param->sharding());
    }
    new_param->set_metadata(ori_param->metadata());    
    old_param_2_new[ori_param] = new_param.get();
    instructions.push_back(std::move(new_param));
  }

  // Process the instructions in the sliced module
  for (auto inst : sliced_instructions) {
    // Map parameter instruction
    if (old_param_2_new.contains(inst)) {
      context->MapInstruction(inst, old_param_2_new[inst]);
    } else {
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* operand : inst->operands()) {
        new_operands.push_back(context->GetInstruction(operand));
      }
      auto new_inst = inst->CloneWithNewOperands(inst->shape(), new_operands, context);
      instructions.push_back(std::move(new_inst));
    }
  }

  // Add tuple instruction as root inst to ret all internal values used by others
  std::vector<HloInstruction*> upd_ret_instructions;
  for (auto inst : sliced_ret_instructions) {
    upd_ret_instructions.push_back(context->GetInstruction(inst));
  }
  if (upd_ret_instructions.size() > 0) {
    auto ret_tuple_inst = HloInstruction::CreateTuple(upd_ret_instructions);  
    instructions.push_back(std::move(ret_tuple_inst));
  }
  

  // Build the HLO computation
  HloComputation::Builder builder(absl::StrCat(entry_comp->name(), "-", sliced_suffix));
  for (auto& inst : instructions) {
    builder.AddInstruction(std::move(inst));
  }
  std::unique_ptr<HloComputation> new_computation = builder.Build();
  // Add control dependency
  for (auto inst : sliced_instructions) {
    HloInstruction* new_inst = context->GetInstruction(inst);
    for (auto successor : inst->control_successors()) {
      TF_CHECK_OK(
          new_inst->AddControlDependencyTo(context->GetInstruction(successor)));
    }
  }

  // NOTE: We assume the HLO graph only has one computation.
  sliced_module->AddEntryComputationWithLayouts(std::move(new_computation));

  return sliced_module;
}

void MemoryOffloader::CollectPartitionCandidates() {
  const HloComputation* entry_comp = fw_module_->entry_computation();
  auto comp_instructions = entry_comp->MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*, int64_t> inst_2_index;

  // Index each instruction
  int64_t counter = 0;
  for (auto inst : comp_instructions) {
    inst_2_index.insert({inst, counter});
    counter++;
  }
  
  // skip the last tuple ret inst
  for (int i = 0; i < comp_instructions.size()-1; i++) {
    auto inst = comp_instructions[i];
    if (inst->opcode() != HloOpcode::kParameter) {
      partition_candidates_.push_back(inst_2_index[inst]);
    }
  }
}

// Find first pos which alloc memory > on chip memory in [slice_start, slice_end) 
int64_t MemoryOffloader::LowerBoundSubGraph(const HloModule* fw_module, int64_t slice_start, int64_t slice_end) {
  auto slice_cnt = slice_end - slice_start;
  int64_t slice_mid = 0;
  int64_t iter_start = slice_start;

  while(slice_cnt > 0) {
    slice_mid = iter_start;
    auto step = slice_cnt / 2;
    slice_mid += step;

    int64_t slice_start_pos = partition_candidates_[slice_start];
    int64_t slice_mid_pos = partition_candidates_[slice_mid];
    // spmd::StdCerr(2) << "Try to Slice hlo module: " << slice_start << ", " 
    //                  << slice_mid << ", " << slice_end
    //                  << ", with pos: " << slice_start_pos << ", " 
    //                  << slice_mid_pos+1 << ", " << partition_candidates_.back()+1 <<  "\n";

    // convert parition candidates index to pos in hlo module first
    // slice module to [slice_start, slice_pos]
    auto sliced_module = SliceHloModule(fw_module, slice_start_pos, slice_mid_pos+1);
    if (sliced_module == nullptr) {
      // fail to slice module occurs
      // [FIXME]: occurs for some single inst param case
      return -1;
    }
    double sliced_alloc_mem = GetHloMoudleAllocMemory(sliced_module.get());
    auto param_info = ParameterAnalysis(sliced_module.get());
    // on chip memory: fw(sub_g_i) * 2 - grads(sub_g_i) + max_grad(sub_g_i)
    double total_mem = sliced_alloc_mem * 2 - param_info.first + param_info.second;
    if (total_mem < debug_minimal_mem_) {
      debug_minimal_mem_ = total_mem;
      minimal_mem_range_.first = slice_start_pos;
      minimal_mem_range_.second = slice_mid_pos+1;
    }
    if (total_mem <= on_chip_memory_) {
      iter_start = ++slice_mid;
      slice_cnt -= step+1;
    } else {
      slice_cnt = step;
    }
    // spmd::StdCerr(2) << " Subgraph Memsize: " << (sliced_alloc_mem/1024/1024/1024) << "GB, " 
    //                                           << (total_mem/1024/1024/1024) << "GB, "
    //                                           << (on_chip_memory_/1024/1024/1024) << "GB" << "\n" ;
    // // spmd::StdCerr(2) << sliced_module->ToString() << "\n";
  }

  // fail to find index can meet the on chip memory
  if (iter_start == slice_start) {
    return -1;
  }

  return iter_start;
}

// Offload strategy 3: subgroups internal res offload
// i.e., layers by layers exectution
// execution flow:
// fw_stage: chip(sub_g_1) -> cpu(internal res) -> chip(sub_g_2) -> ... -> chip(sub_g_n)
// bw stage: in addtition to grads offlaod, do same partition as fw_stage
void MemoryOffloader::OffloadViaStrategy3() {
  // first get valid partition insts
  CollectPartitionCandidates();
  // if no valid partition candidates
  if (partition_candidates_.size() == 0) {
    fw_offload_cost_ = 100;
    bw_offload_cost_ = 100;
    apply_grad_offload_cost_ = 100;
    return;
  }
  // add 0, end to parition cadidates
  const HloComputation* comp = fw_module_->entry_computation();
  // slice_start, slice_end are variables related to binary search algorithm
  // not the index in hlo module, hlo module index comes from partition_candidates_
  int64_t slice_start = 0;
  int64_t slice_end = partition_candidates_.size();
  // spmd::StdCerr(2) << "\n\nStart subgraph partition: " << slice_start << " ---> " << slice_end 
  //                 << ", with pos: " << partition_candidates_[slice_start] << " ---> " << partition_candidates_.back() << "\n";
  debug_minimal_mem_ = 1e20;
  slice_start = LowerBoundSubGraph(fw_module_, slice_start, slice_end);
  while(slice_start < slice_end) {
    if (slice_start < 0) {
      break;
    }  
    slice_pos_.push_back(partition_candidates_[slice_start]);
    // spmd::StdCerr(2) << "Partition point: " << slice_start << ", with pos: " << partition_candidates_[slice_start] << "\n" ;
    debug_minimal_mem_ = 1e20;
    slice_start = LowerBoundSubGraph(fw_module_, slice_start, slice_end);
  }

  // if last slice_start is -1, means fail to find minimal subgraph
  // ret relative large cost 
  if (slice_start == -1) {
    slice_pos_.push_back(-1); // just for log
    fw_offload_cost_ = 100;
    bw_offload_cost_ = 100;
    apply_grad_offload_cost_ = 100;
    return;
  }
  // No need partition hlo module, return strategy 1 cost 
  // may not enter in estimation process, a test branch
  if (slice_pos_.empty()) {
    fw_offload_cost_ = 0;
    bw_offload_cost_ = grads_size_ / memory_bandwidth_;
    apply_grad_offload_cost_ = params_size_ / memory_bandwidth_;
    return;
  }

  // Construct partitioned modules sizeof(slice_pos) + 1
  auto pos_num = slice_pos_.size();
  int64_t slice_pos_start = partition_candidates_[0];
  int64_t slice_pos_end = partition_candidates_.back();
  std::vector<std::unique_ptr<HloModule>> slice_modules;
  // [pos_start, pos_0)
  if (slice_pos_start != slice_pos_[0]) {
    auto slice_module = SliceHloModule(fw_module_, slice_pos_start, slice_pos_[0]); 
    slice_modules.push_back(std::move(slice_module));
  }
  // [pos_0, pos_1) ... [pos_last-1, pos_last) etc.
  for (int i = 0; i < pos_num - 1; i++) {
    auto slice_module = SliceHloModule(fw_module_, slice_pos_[i], slice_pos_[i+1]); 
    slice_modules.push_back(std::move(slice_module));
  }
  // [pos_last, pos_end)
  if (slice_pos_[pos_num-1] != slice_pos_end) {
    auto slice_module = SliceHloModule(fw_module_, slice_pos_[pos_num-1], slice_pos_end);  
    slice_modules.push_back(std::move(slice_module));
  }
  
  // Get each subgroup module infos
  double total_in_out_params = 0;
  for (int i = 0; i < slice_modules.size(); i++) {
    const HloModule* module = slice_modules[i].get();
    double sub_mem = GetHloMoudleAllocMemory(module);
    auto in_out_param = ParameterAnalysis(module);
    total_in_out_params += in_out_param.first;
    in_out_params_.push_back(in_out_param.first);

    // on chip memory: fw(sub_g_i) * 2 - grads(sub_g_i) + max_grad(sub_g_i)
    double sub_total_mem = sub_mem * 2 - in_out_param.first + in_out_param.second;
    slice_mems_.push_back(sub_total_mem);
  }

  // fw_offload_cost_: total(sub_groups in/out)
  // bw_offload_cost_: (grads) + total(sub_groups in/out)
  // apply_grad_offload_cost_: (params)
  fw_offload_cost_ = total_in_out_params / memory_bandwidth_;
  bw_offload_cost_ = (total_in_out_params + grads_size_) / memory_bandwidth_;
  apply_grad_offload_cost_ = params_size_ / memory_bandwidth_;

  return;
}

// ------ Entry func to estimate offload cost ------
void MemoryOffloader::EstimateOffloadCost() {
  // Initial all cost as 0s
  fw_offload_cost_ = 0;
  bw_offload_cost_ = 0;
  apply_grad_offload_cost_ = 0;

  // parameter analysis
  auto param_info = ParameterAnalysis(fw_module_);
  params_size_ = param_info.first;
  grads_size_ = params_size_;
  max_grad_size_ = param_info.second;

  // Estimate modules alloc memory
  fw_alloc_memory_ = GetHloMoudleAllocMemory(fw_module_);
  bw_alloc_memory_= GetHloMoudleAllocMemory(bw_module_);
  apply_grad_alloc_memory_= GetHloMoudleAllocMemory(apply_grad_module_);

  // if exists large param out of on_chip_mem, return large cost as penalty
  if (max_grad_size_ > on_chip_memory_) {
    fw_offload_cost_ = 100;
    bw_offload_cost_ = 100;
    apply_grad_offload_cost_ = 100;
    LoggingMemoryInfos(0);
    return;
  }
  
  // If can fit all modules alloc memory
  total_alloc_mem_ = fw_alloc_memory_ + bw_alloc_memory_ + apply_grad_alloc_memory_;
  if (total_alloc_mem_ <= on_chip_memory_) {
    LoggingMemoryInfos(1);
    return;
  }

  // Offload strategy 1: opt states offload
  // execution flow:
  // opt stage: chip(grads) ---> cpu(udpated params) ---> chip 
  // result:
  // on chip memory: (fw + bw)
  // apply_grad_offload_cost_: (grads + params)
  strategy_1_alloc_mem_ = fw_alloc_memory_ + bw_alloc_memory_;
  if (strategy_1_alloc_mem_ <= on_chip_memory_) {
    apply_grad_offload_cost_ = (params_size_ + grads_size_) / memory_bandwidth_;    
    LoggingMemoryInfos(2);
    return;
  }

  // Offload strategy 2: bw grads offload
  // execution flow:
  // bw stage: chip(grads) ---> cpu(updated params)
  // result:
  // on chip memory: (fw + bw - grads_size + max_grad_size)
  // bw_offload_cost_: (grads)
  // apply_grad_offload_cost_: (params)
  strategy_2_alloc_mem_ = fw_alloc_memory_ + bw_alloc_memory_ - grads_size_ + max_grad_size_;
  if (strategy_2_alloc_mem_ <= on_chip_memory_) {
    // assume (save grad & bw comp) overlap, (load grad & grad upd) overlap
    bw_offload_cost_ = grads_size_ / memory_bandwidth_;
    apply_grad_offload_cost_ = params_size_ / memory_bandwidth_;
    LoggingMemoryInfos(3);
    return;
  }

  // Offload strategy 3, layers by layers execution
  // strategy 3 mem alloc & cost obtained in internal func
  OffloadViaStrategy3();
  LoggingMemoryInfos(4);
  return;
}

void MemoryOffloader::LoggingMemoryInfos(int type) const {
  auto toGB = [](double size) {
    return size/1024/1024/1024;
  };

  std::ostringstream os;
  std::vector<std::string> log_types = {"super_large_param", "whole_module_can_fit", 
                                        "offload_strategy_1", "offload_strategy_2", 
                                        "offload_strategy_3"};

  os << "\n------------ Memory Offload Alloc Infos Start ------------\n";

  os << "On chip memory: " << toGB(on_chip_memory_) 
                           << " GB, memory bw: " << toGB(memory_bandwidth_) << " GB/s" << "\n"; 
  os << "Hlo module alloc infos: \n" 
                           << "\t\tfw module: " << toGB(fw_alloc_memory_) << " GB\n"
                           << "\t\tbw module: " << toGB(bw_alloc_memory_) << " GB\n"
                           << "\t\tapply_grad module: " << toGB(apply_grad_alloc_memory_) << " GB\n"
                           << "\t\ttotal: " << toGB(total_alloc_mem_) << " GB\n";
  os << "Params infos: \n" 
                  << "\t\tparams(grads): " << toGB(params_size_) << " GB\n"
                  << "\t\tmax grad size: " << toGB(max_grad_size_) << " GB\n";
  os << "Choose strategy: " << log_types[type] << "\n";

  os << "Each stage cost: \n" 
                  << "\t\tfw offload cost: " << fw_offload_cost_ << "\n"
                  << "\t\tbw offload cost: " << bw_offload_cost_ << "\n"
                  << "\t\tapply grad offload cost: " << apply_grad_offload_cost_ << "\n";
  
  if (type == 2) {
    // strategy 1
    os << "Strategy 1 alloc info: " << toGB(strategy_1_alloc_mem_) << " GB\n";
  } else if (type == 3) {
    // strategy 2
    os << "Strategy 2 alloc info: " << toGB(strategy_2_alloc_mem_) << " GB\n";
  } else if (type == 4) {
    // strategy 3
    os << "Strategy 3 alloc infos: \n";
    os << "\t\tfW module length: " << fw_module_->entry_computation()->instruction_count() << "\n";
    os << "\t\tpartition candidates size: " << partition_candidates_.size() - 1 << "\n";
    // slice pos infos 
    os << "\t\tpos: [" << partition_candidates_[0] << " ";
    for (auto pos : slice_pos_) {
      os << pos << " ";
    }
    os << partition_candidates_.back() << "]" <<  "\n";
    // in out params infos
    os << "\t\tin/out param size(GB): ";
    for (auto param : in_out_params_) {
      os << toGB(param) << " ";
    }
    os << "\n";
    // sliced modules infos
    os << "\t\tsliced module mem(GB): ";
    for (auto mem : slice_mems_) {
      os << toGB(mem) << " ";
    }
    os << "\n";
    if (std::find(slice_pos_.begin(), slice_pos_.end(), -1) != slice_pos_.end()) {
      os << "\t\tdebug minimal subgroup mem usage: " << toGB(debug_minimal_mem_) << " GB, range:[" 
      << minimal_mem_range_.first << ", " << minimal_mem_range_.second << ")\n";
    }
    // total in/out param size
    double total_param_size = std::accumulate(in_out_params_.begin(), in_out_params_.end(), 0.0);
    os << "\t\ttotal in/out param: " << toGB(total_param_size) << " GB\n"; 
  }

  os << "------------ Memory Offload Alloc Infos End ------------\n\n";
  spmd::StdCerr(2) << os.str();
}

std::vector<double> AnalyticalMemoryCostOfHloModule(const HloModule* fw_module, const HloModule* bw_module, 
                                                    const HloModule* apply_grad_module) {
  bool force_use_fp16 = pass_context::GetBool("analytical_perf::force_use_fp16", false);
  std::string hardware = pass_context::GetString("analytical_perf::hardware", "gpu");

  // hardware == "gpu"
  int64_t card_mem = pass_context::GetInt("analytical_perf_gpu::card_mem", pow(10, 9));
  int64_t gpu_pcie_bandwidth = pass_context::GetInt("analytical_perf_gpu::pcie_bandwidth", pow(10, 9));
  // hardware == "wsc"
  int64_t ddr_mem = pass_context::GetInt("analytical_perf_wsc::ddr_mem", pow(10, 9));
  int64_t wsc_pcie_bandwidth = pass_context::GetInt("analytical_perf_wsc::pcie_bandwidth", pow(10, 9));

  MemoryOffloader mem_offloader(fw_module, bw_module, apply_grad_module, force_use_fp16);

  if (hardware == "gpu") {
    mem_offloader.SetHardwareConfigs(card_mem, gpu_pcie_bandwidth);
  } else if (hardware == "wsc") {
    mem_offloader.SetHardwareConfigs(ddr_mem, wsc_pcie_bandwidth);
  }
  mem_offloader.EstimateOffloadCost();

  auto cost = mem_offloader.GetStagesCost(); 
  return cost;
}

}  // namespace gpu
}  // namespace xla
