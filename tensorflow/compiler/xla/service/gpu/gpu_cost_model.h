#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_

#include <pybind11/pybind11.h>
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace gpu {

namespace py = pybind11;

double EstimateHloModuleCost(const HloModule* hlo_module);

StatusOr<double> EstimateHloModuleMemory(HloModule* hlo_module);

struct NetworkDescrib
{
    std::string comm_type;
    std::string replica_groups_str;
    int comm_data_count;
    int data_type;
    void set_value(std::string comm_type_p, std::string replica_groups_str_p, int comm_data_count_p, int data_type_p){
        comm_type = comm_type_p;
        replica_groups_str = replica_groups_str_p;
        comm_data_count = comm_data_count_p;
        data_type = data_type_p;
    }
};

struct HloCost{
    std::string op_name;
    double flops;
    NetworkDescrib network_describ;
    int data_type;
    double network_count;
    double estimated_time;
    void set_value(std::string op_name_p, double flops_p, std::string comm_type_p, 
                   std::string replica_groups_str_p, int comm_data_count_p, int data_type_p, 
                   double network_count_p, double estimated_time_p=0.0){
        op_name = op_name_p;
        flops = flops_p;
        network_describ.set_value(comm_type_p, replica_groups_str_p, comm_data_count_p, data_type_p);
        network_count = network_count_p;
        data_type = data_type_p;
        estimated_time = estimated_time_p;
    }
    void set_flops(std::string op_name_p, double flops_p, int data_type_p) {
        op_name = op_name_p;
        flops = flops_p;
        data_type = data_type_p;
        network_count = 0.0;
    }
    void set_flops_network(std::string op_name_p, double flops_p, int data_type_p, double network_count_p, 
                           double estimated_time_p=0.0) {
        op_name = op_name_p;
        flops = flops_p;
        data_type = data_type_p;  
        network_count = network_count_p;
        estimated_time = estimated_time_p;
    }
};

py::list HloModuleCost(const HloModule* hlo_module);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_
