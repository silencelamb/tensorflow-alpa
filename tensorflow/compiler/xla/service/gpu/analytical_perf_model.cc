#include <bits/stdint-intn.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <math.h>
#include <cstdint>
#include <memory>
#include <numeric>
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

    if (ins->IsCustomCall(xla::gpu::kGemmCallTarget)) {
      const HloInstruction* lhs = ins->operand(0);
      const HloInstruction* rhs = ins->operand(1);
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      auto config = ins->backend_config<xla::gpu::GemmBackendConfig>().ValueOrDie();
      auto dnum = config.dot_dimension_numbers();
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
double MemoryOffloader::ParameterAnalysis(const HloModule* module) {
  double max_param = 0.0;
  const HloComputation* entry_comp = module->entry_computation();
  for (auto inst : entry_comp->parameter_instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      auto shape = inst->shape();
      if (force_use_fp16_) 
        shape.set_element_type(PrimitiveType::F16);
      if (shape.is_dynamic())
        continue;
      max_param = std::max(spmd::GetBytes(shape), max_param);
    }
  }
  return max_param;
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

// ------ Offload Strategy 1 ------
void MemoryOffloader::OffloadViaStrategy1() {
  const HloComputation* fw_comp = fw_module_->entry_computation();
  auto fw_instructions = fw_comp->MakeInstructionPostOrder();
  bool force_use_fp16 = force_use_fp16_;
  auto calc_shape_bytes = [&force_use_fp16](Shape shape){
    if (force_use_fp16) {
      shape.set_element_type(PrimitiveType::F16);
    }
    return spmd::GetBytes(shape);
  };
  offload_1_alloc_memory_ = 0;

  // [REMOVE ME]: Previous single graph memory model dev code
  // // 1) Find grads related params, i.e., first use after op barrier inst
  // // Index each instruction, find opt barrier instruction index
  // int64_t counter = 0;
  // int64_t opt_barrier_idx = 0;
  // absl::flat_hash_map<const HloInstruction*, int64_t> inst_2_index;
  // for (auto inst : comp_instructions) {
  //   inst_2_index.insert({inst, counter});
  //   if (inst->opcode() == HloOpcode::kOptimizationBarrier) {
  //     opt_barrier_idx = counter;
  //   }
  //   counter++;
  // }

  // // Filter out the grads instructions
  // max_grad_size_ = 0;
  // std::vector<const HloInstruction*> grad_params;
  // int64_t bw_first_partition_idx = INT64_MAX;
  // bool use_opt_barrier_partition = true;

  // if (use_opt_barrier_partition) {
  //   // [TODO]: opt_barrier method may fail for some cases 
  //   // i.e., can't find param inst whose first use > opt_barrier_idx
  //   if (opt_barrier_idx == 0) {
  //     // Fail to find opt barrier
  //     offload_1_alloc_memory_ = 0;
  //     return;
  //   }
  //   bw_first_partition_idx = opt_barrier_idx;
  // } else {
  //   // obtain infos about dot, conv ops. these ops must have same number in fw & bw
  //   // use mid inst as bw start
  //   // [FIXME]: not feasible, bw may have more ops, still need a partition point?
  //   std::vector<const HloInstruction*> heavy_ops;
  //   for (auto inst : comp_instructions) {
  //     auto op_code = inst->opcode();
  //     if (op_code == HloOpcode::kDot || op_code == HloOpcode::kConvolution) {
  //       heavy_ops.push_back(inst);
  //     }
  //   }
  //   if (heavy_ops.size() < 1) {
  //     offload_1_alloc_memory_ = 0;
  //     return;
  //   }
  //   const HloInstruction* bw_first_heavy_inst = heavy_ops[heavy_ops.size() / 2];
  //   CHECK(inst_2_index.contains(bw_first_heavy_inst));
  //   bw_first_partition_idx = inst_2_index[bw_first_heavy_inst];
  // }
    
  // use param in fw to caculate size of grads, 1-to-1
  offload_grads_size_ = 0;
  for (auto inst : fw_instructions) {
    if (inst->opcode() == HloOpcode::kParameter) {
      // save maximum grad size for future use
      max_grad_size_ = std::max(max_grad_size_, calc_shape_bytes(inst->shape()));
      offload_grads_size_ += calc_shape_bytes(inst->shape()); 
    } 
  }


  // estimates fw memory + 2*max(grad) + max(comp_op operands)
  max_act_size_ = 0;
  // auto fw_module = SliceHloModule(comp_module_, 0, bw_first_partition_idx);
  for (auto inst : bw_module_->entry_computation()->instructions()) {
    // enter bw graph
    auto op_code = inst->opcode();
    if (op_code == HloOpcode::kDot || op_code == HloOpcode::kConvolution) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      // save maximum act size for future use, here assume no diff in op operands
      max_act_size_ = std::max(max_act_size_, std::max(calc_shape_bytes(lhs->shape()), 
                                                    calc_shape_bytes(rhs->shape())));
    }
  }
  offload_1_alloc_memory_ += fw_alloc_memory_;
  offload_1_alloc_memory_ += (bw_alloc_memory_ - offload_grads_size_ + max_grad_size_);
}

// Find first pos which alloc memory > on chip memory in [slice_start, slice_end) 
int64_t MemoryOffloader::LowerBoundSubGraph(const HloModule* fw_module, int64_t slice_start, int64_t slice_end) {
  auto slice_cnt = slice_end - slice_start;
  int64_t slice_pos = 0;
  int64_t iter_start = slice_start;

  while(slice_cnt > 0) {
    slice_pos = iter_start;
    auto step = slice_cnt / 2;
    slice_pos += step;

    // spmd::StdCerr(2) << "Try to Slice hlo module: " << slice_start << ", " 
    //                  << (slice_pos+1) << ", " << slice_end << "\n";

    // slice module to [slice_start, slice_pos]
    auto sliced_module = SliceHloModule(fw_module, slice_start, slice_pos+1);
    if (sliced_module == nullptr) {
      // fail to slice module occurs
      // [FIXME]: occurs for some single inst case
      return -1;
    }
    double sliced_alloc_mem = GetHloMoudleAllocMemory(sliced_module.get());
    double total_mem = sliced_alloc_mem + 2*max_grad_size_ + max_act_size_;
    minimal_mem_ = std::min(total_mem, minimal_mem_);
    if (total_mem <= on_chip_memory_) {
      iter_start = ++slice_pos;
      slice_cnt -= step+1;
    } else {
      slice_cnt = step;
    }
    // spmd::StdCerr(2) << " Subgraph Memsize: " << (sliced_alloc_mem/1024/1024/1024) << "GB, " 
    //                                           << (total_mem/1024/1024/1024) << "GB, "
    //                                           << (minimal_mem_/1024/1024/1024) << "GB, "
    //                                           << (on_chip_memory_/1024/1024/1024) << "GB" << "\n" ;
    // spmd::StdCerr(2) << sliced_module->ToString() << "\n";
  }

  // fail to find index can meet the on chip memory
  if (iter_start == slice_start) {
    return -1;
  }

  return iter_start;
}

// ------ Offload Strategy 2 ------
double MemoryOffloader::OffloadViaStrategy2() {
  const HloComputation* comp = fw_module_->entry_computation();

  int64_t slice_start = 0;
  int64_t slice_end = comp->instruction_count();

  // spmd::StdCerr(2) << "\n\nStart subgraph partition: " << slice_start << " ---> " << slice_end << "\n" ;

  slice_start = LowerBoundSubGraph(fw_module_, slice_start, slice_end);
  while(slice_start < slice_end) {
    if (slice_start < 0) {
      break;
    }
    slice_pos_.push_back(slice_start);
    // spmd::StdCerr(2) << "Partition point: " << slice_start << "\n" ;
    slice_start = LowerBoundSubGraph(fw_module_, slice_start, slice_end);
  }

  // if last slice_start is -1, means fail to find minimal subgraph
  // ret relative large cost 
  if (slice_start == -1) {
    slice_pos_.push_back(-1); // just for log
    return 100;
  }
  // No need partition hlo module, return strategy 1 cost 
  // may not enter in estimation process, a test branch
  if (slice_pos_.empty()) {
    return offload_grads_size_ / memory_bandwidth_;
  }

  bool force_use_fp16 = force_use_fp16_;
  auto GetParamSize = [&force_use_fp16](const HloModule* local_module) {
    double params_size = 0;
    const HloComputation* entry_comp = local_module->entry_computation();
    for (auto inst : entry_comp->instructions()) {
      if (inst->opcode() == HloOpcode::kParameter) {
        auto shape = inst->shape();
        if (force_use_fp16) {
          shape.set_element_type(PrimitiveType::F16);
        }
        if (shape.is_dynamic())
          continue;
        params_size += spmd::GetBytes(shape);
      }
    }
    return params_size;
  };

  // Construct partitioned modules sizeof(slice_pos) + 1
  double total_in_out_params = 0;
  double in_out_param = 0;
  double offload_cost = 0;
  auto pos_num = slice_pos_.size();
  // [0, pos_0)
  auto slice_module = SliceHloModule(fw_module_, 0, slice_pos_[0]); 
  // [pos_0, pos_1) ... [pos_last-1, pos_last) etc.
  for (int i = 0; i < pos_num - 1; i++) {
    slice_module = SliceHloModule(fw_module_, slice_pos_[i], slice_pos_[i+1]); 
    // add param size 
    in_out_param = GetParamSize(slice_module.get());
    in_out_params_.push_back(in_out_param);
    total_in_out_params += in_out_param;
  }
  // [pos_last, slice_end)
  slice_module = SliceHloModule(fw_module_, slice_pos_[pos_num-1], slice_end);  
  in_out_param = GetParamSize(slice_module.get());
  in_out_params_.push_back(in_out_param);
  total_in_out_params += in_out_param;

  // cost of strategy_2 = cost of strategy_1 + subgraph in/out load/save cost
  offload_cost = (2*total_in_out_params + offload_grads_size_) / memory_bandwidth_;
  return offload_cost;
}

// ------ Entry func to estimate offload cost ------
double MemoryOffloader::EstimateOffloadCost() {
  // First do parameter analysis
  auto largest_param = ParameterAnalysis(fw_module_);
  if (largest_param > on_chip_memory_) {
    offload_cost_ = 100;
    LoggingMemoryInfos(0);
    return offload_cost_;
  }

  // Estimate whole module alloc memory
  fw_alloc_memory_ = GetHloMoudleAllocMemory(fw_module_);
  bw_alloc_memory_= GetHloMoudleAllocMemory(bw_module_);
  apply_grad_alloc_memory_= GetHloMoudleAllocMemory(apply_grad_module_);

  if ((fw_alloc_memory_ + bw_alloc_memory_ + apply_grad_alloc_memory_) <= on_chip_memory_) {
    offload_cost_ = 0;
    LoggingMemoryInfos(1);
    return offload_cost_;
  }

  // Offload strategy 1
  OffloadViaStrategy1();
  if (offload_1_alloc_memory_ <= on_chip_memory_) {
    // assume (save grad & bw comp) overlap, (load grad & grad upd) overlap
    // thus, main cost from load grad back to on chip mem
    offload_cost_ = offload_grads_size_ / memory_bandwidth_;
    LoggingMemoryInfos(2);
    return offload_cost_;
  }

  // Offload strategy 2
  offload_cost_ = OffloadViaStrategy2();
  LoggingMemoryInfos(3);

  return offload_cost_;
}

void MemoryOffloader::LoggingMemoryInfos(int type) {
  auto toGB = [](double size) {
    return size/1024/1024/1024;
  };

  std::ostringstream os;
  std::vector<std::string> log_types = {"super_large_param", "whole_module_can_fit", 
                                        "offload_strategy_1", "offload_strategy_2"};

  os << "\n------------ Memory Offload Alloc Infos Start ------------\n";

  os << "On chip memory: " << toGB(on_chip_memory_) 
                           << " GB, memory bw: " << toGB(memory_bandwidth_) << " GB/s" << "\n"; 
  os << "Largest parameter size: " << toGB(ParameterAnalysis(fw_module_)) << " GB\n";
  os << "Hlo module alloc infos: \n" 
                           << "\t\tfw module: " << toGB(fw_alloc_memory_) << " GB\n"
                           << "\t\tbw module: " << toGB(bw_alloc_memory_) << " GB\n"
                           << "\t\tapply_grad module: " << toGB(apply_grad_alloc_memory_) << " GB\n"
                           << "\t\ttotal: " << toGB(fw_alloc_memory_+bw_alloc_memory_+apply_grad_alloc_memory_) << " GB\n";

  os << "Offload Strategy 1 memory infos: \n" 
                   << "\t\ttotal: " << toGB(offload_1_alloc_memory_) << " GB\n"
                   << "\t\tbw subtract way: " << toGB(bw_alloc_memory_-offload_grads_size_+max_grad_size_) << " GB\n"
                   << "\t\testimate way: " << toGB(2*max_grad_size_ + max_act_size_) << " GB\n"
                   << "\t\tmax grad size: " << toGB(max_grad_size_) << " GB\n"
                   << "\t\toffload grads size: "<< toGB(offload_grads_size_) << " GB\n";
  
  os << "Offload Strategy 2 memory infos: \n"
                   << "\t\tfw module length: " << fw_module_->entry_computation()->instruction_count() << "\n"
                   << "\t\tslice_pos size: " << slice_pos_.size() << "\n"
                   << "\t\tminimal mem among all partitions: " << toGB(minimal_mem_) << " GB\n";
  // slice pos infos 
  os << "\t\tpos: ";
  for (auto pos : slice_pos_) {
    os << pos << " ";
  }
  os << "\n";
  // in out params infos
  os << "\t\tin/out param size(GB): ";
  for (auto param : in_out_params_) {
    os << toGB(param) << " ";
  }
  os << "\n";
  // total in/out param size
  double total_param_size = std::accumulate(in_out_params_.begin(), in_out_params_.end(), 0.0);
  os << "\t\ttotal in/out param: " << toGB(total_param_size) << " GB\n";  
  
  // Final offload cost
  os << "Final Offload Cost From [" << log_types[type] << "] : " << offload_cost_ << "\n";  
  os << "------------ Memory Offload Alloc Infos End ------------\n";

  spmd::StdCerr(2) << os.str();
}

double AnalyticalMemoryCostOfHloModule(const HloModule* fw_module, const HloModule* bw_module, 
                                      const HloModule* apply_grad_module) {
  bool force_use_fp16 = pass_context::GetBool("analytical_perf::force_use_fp16", false);
  std::string hardware = pass_context::GetString("analytical_perf::hardware", "gpu");
  double cost = 0;

  // hardware == "gpu"
  int64_t card_mem = pass_context::GetInt("analytical_perf_gpu::card_mem", pow(10, 9));
  int64_t gpu_ddr_bandwidth = pass_context::GetInt("analytical_perf_gpu::ddr_bandwidth", pow(10, 9));
  // hardware == "wsc"
  int64_t tile_mem = pass_context::GetInt("analytical_perf_wsc::tile_mem", pow(10, 9));
  int64_t wsc_ddr_bandwidth = pass_context::GetInt("analytical_perf_wsc::ddr_bandwidth", pow(10, 9));

  MemoryOffloader mem_offloader(fw_module, bw_module, apply_grad_module, force_use_fp16);

  if (hardware == "gpu") {
    mem_offloader.SetHardwareConfigs(card_mem, gpu_ddr_bandwidth);
  } else if (hardware == "wsc") {
    mem_offloader.SetHardwareConfigs(tile_mem, wsc_ddr_bandwidth);
  }
  cost = mem_offloader.EstimateOffloadCost();

  return cost;
}

}  // namespace gpu
}  // namespace xla
