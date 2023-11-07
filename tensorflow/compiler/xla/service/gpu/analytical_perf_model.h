#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ANALYTICAL_PERF_MODEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ANALYTICAL_PERF_MODEL_H_

#include <pybind11/pybind11.h>
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace analytical_perf {
    enum COMM_MODE {
        ALL_GATHER=0,
        REDUCE_SCATTER=1,
        ALL_REDUCE=2,
        ALL_TO_ALL=3,
    };

    inline double get_comm_total_size(int64_t comm_bytes, COMM_MODE comm_mode, int64_t num) {
        // 根据要通信的数据，得出num个设备之间的通信总量
        int64_t total_size = 0;
        switch (comm_mode) {
            case ALL_GATHER:
                total_size = comm_bytes * (num-1);
                break;

            case REDUCE_SCATTER:
                total_size = comm_bytes * (num-1);
                break;

            case ALL_REDUCE:
                total_size = comm_bytes * (num-1) * 2;
                break;

            case ALL_TO_ALL:
                total_size = comm_bytes * (num-1);
                break;

            default:
                break;
        }
        return total_size;
        
    }
    class BaseUnit {
    /*
    Abstract base class, based on the unit, hardware hierarchy is like this:
    GPU：Card(BaseUnit) -> Node(e.g. 8 Cards)   -> Rack                -> ...
    WSC: Tile(BaseUnit) -> Die (e.g. 6x6 Tiles) -> WSC (e.g. 6x6 Dies) -> ...
    */
        public:
            std::map<PrimitiveType, int64_t> compute;
            int64_t band_width; // Bytes per sencond
            int64_t local_mem; // Bytes
            double compute_utilization; // 0~1
            double bandwidth_utilization; // 0~1

            virtual double AnalyseComputeTime(int64_t ops_num, PrimitiveType datatype, int64_t unit_num, bool force_use_fp16) {
                if (force_use_fp16) {
                    datatype = F16;
                }
                return ops_num / (compute[datatype] * unit_num * compute_utilization);
            }
            virtual double AnalyseCommunicateTime(int64_t comm_bytes, COMM_MODE comm_mode, int64_t unit_num) {
                int64_t total_size = get_comm_total_size(comm_bytes, comm_mode, unit_num);
                return total_size / (band_width * unit_num * bandwidth_utilization);
            }
            BaseUnit(std::map<PrimitiveType, int64_t> cmp={}, int64_t bw=0, int64_t mem=0, double cmp_ul=1.0, double bw_ul=1.0) {
                compute = cmp;
                band_width = bw;
                local_mem = mem;
                compute_utilization = cmp_ul;
                bandwidth_utilization = bw_ul;
            }
    };

    namespace gpu {
    // Analytical Performance Model for GPU
    // Card(BaseUnit) -> Node(e.g. 8 Cards) -> Rack -> ...
        class Card: public BaseUnit {
            public:
                using BaseUnit::BaseUnit;
        };
        class Node {
            // node之间有bandwidth，node没有单独自己的local_mem
            public:
                std::vector <Card> cards; // 由外面设置
                int64_t node_band_width;      // 节点间的通信带宽
                double node_bw_ul;         // 节点间通信带宽利用率

                void SetCardConfig();
                
                Node(int64_t card_num=8, std::map<PrimitiveType, int64_t> card_cmp={}, int64_t card_bw=0, int64_t card_mem=0, 
                        int64_t node_bw=0, double cmp_ul=1.0, double bw_ul=1.0) {
                    for(int64_t i = 0; i < card_num; i++){
                        cards.push_back(Card(card_cmp, card_bw, card_mem, cmp_ul, bw_ul));
                    }
                    node_band_width = node_bw;
                    node_bw_ul = bw_ul;
                }
                
                double AnalyseComputeTime(int64_t ops_num, PrimitiveType datatype, int64_t node_num, bool force_use_fp16) {
                    if (force_use_fp16) {
                        datatype = F16;
                    }
                    double total_compute = node_num * cards.size() * cards[0].compute[datatype] * 
                        cards[0].compute_utilization;  // 先认为每个卡的利用率是一样的
                    return ops_num / total_compute;
                }
                
                double AnalyseCommunicateTime(int64_t comm_bytes, COMM_MODE comm_mode, int64_t node_num, int verbose) {
                    // 跨了节点时通信性能估计的策略
                    // 初步策略，节点间通信的时间 与 节点内通信的时间, 两者的最大值
                    int64_t card_num = cards.size();
                    double intra_node_time = cards[0].AnalyseCommunicateTime(comm_bytes, comm_mode, card_num);
                    int64_t total_size_inter_node = get_comm_total_size(comm_bytes, comm_mode, node_num);
                    double inter_node_time = total_size_inter_node / (node_band_width * node_num * node_bw_ul);
                    if (verbose == 2) {
                        std::cout << std::endl <<  "intra_node_time: " << intra_node_time 
                                << ", total_size_inter_node: " << total_size_inter_node
                                << ", inter_node_time: " << inter_node_time << std::endl;
                    }
                    return std::max(intra_node_time, inter_node_time);
                }
        };
        
    }
    namespace wsc { 
    // Analytical Performance Model for Wafe Scale Chip
    // Tile(BaseUnit) -> Die (e.g. 6x6 Tiles) -> WSC (e.g. 6x6 Dies) -> ...
        class Tile: public BaseUnit {
            public:
                using BaseUnit::BaseUnit;
                double AnalyseCommunicateTime(int64_t comm_bytes, COMM_MODE comm_mode, 
                            int64_t tile_r_num, int64_t tile_c_num) {
                    // 矩形的tile阵列
                    // 后面也许可以先各自ring，然后再同步？ 层次ring
                    // TODO: 需分析对比一下看看
                    int64_t total_tile_num = tile_r_num * tile_c_num;
                    int64_t total_size = get_comm_total_size(comm_bytes, comm_mode, total_tile_num);
                    return total_size / (band_width * total_tile_num * bandwidth_utilization);
            }
        };

        class Die {
            //die之间有bandwidth，die没有单独自己的local_mem
            public:
                std::vector<std::vector<Tile>> tiles; // 由外面设置;
                double die_band_width;      // Die间的通信带宽
                double die_bw_ul;         // Die间通信带宽利用率
                double die_alpha;    // Link lateny between two dies, measured in s, NVLink is 0.7 * 10e-6 s
                double die_beta;     // s/Byte
                
                void SetTileConfig();

                Die(int64_t tile_r_num=6, int64_t tile_c_num=6, std::map<PrimitiveType, int64_t> tile_cmp={}, int64_t tile_bw=0, double tile_mem=0.0, 
                        double die_bw=0.0, double cmp_ul=1.0, double bw_ul=1.0, double alpha=1.0, double beta=1.0) {
                    for(int64_t i = 0; i < tile_r_num; i++){
                        std::vector<Tile> tiles_one_row;
                        for(int64_t j = 0; j < tile_c_num; j++) {
                            tiles_one_row.push_back(Tile(tile_cmp, tile_bw, tile_mem, cmp_ul, bw_ul));
                        }
                        tiles.push_back(tiles_one_row);
                    }
                    die_band_width = die_bw;
                    die_bw_ul = bw_ul;
                    die_alpha = alpha;
                    die_beta = beta;
                }
                double AnalyseComputeTime(int64_t ops_num, PrimitiveType datatype ,int64_t die_num, bool force_use_fp16) {
                    if (force_use_fp16) {
                        datatype = F16;
                    }
                    double total_compute = die_num * tiles.size() * tiles[0].size() * tiles[0][0].compute[datatype] * 
                        tiles[0][0].compute_utilization;  // 先认为每个卡的利用率是一样的
                    return ops_num / total_compute;                    
                }

                double AnalyseCommunicateTime(int64_t comm_bytes, COMM_MODE comm_mode, int64_t die_num) {
                    // 跨Die通信性能估计的策略
                    // 初步策略，Die间通信的时间 与 Die内通信的时间, 两者的最大值
    
                    double intra_die_time = tiles[0][0].AnalyseCommunicateTime(comm_bytes, comm_mode,
                         tiles.size(), tiles[0].size());
                    int64_t total_size_inter_die = get_comm_total_size(comm_bytes, comm_mode, die_num);
                    double inter_die_time = total_size_inter_die / (die_band_width * die_num * die_bw_ul);
                    return std::max(intra_die_time, inter_die_time);
                }

                double AnalyseCommunicateTime(double comm_bytes, COMM_MODE comm_mode, int64_t die_r_num, int64_t die_c_num) {
                    // 跨Die通信性能估计的策略
                    // 如果是矩形的输入，先换成总的die数目
                    int64_t total_die_num = die_r_num * die_c_num;
                    return AnalyseCommunicateTime(comm_bytes, comm_mode, total_die_num);
                }
                // Time = TimeSteps * (alpha + comm_bytes * beta)
                double AnalyseCommunicateTimeGreedy(int64_t comm_bytes, COMM_MODE comm_mode, int64_t die_num, int time_steps) {
                    return time_steps * (die_alpha + comm_bytes * die_beta);
                }
        };

        class WafeScaleChip {
            // Wafe scale chip 之间互联bandwidth
            public:
                // Die[6][6];   // 计算资源
                double AnalyseComputeTime(double ops_num, int64_t wsc_num) { 
                    return 0.0; 
                }
                double AnalyseCommunicateTime(double comm_bytes, COMM_MODE comm_mode, int64_t wsc_num) { 
                    return 0.0; 
                }
        };
    }
    double AnalyticalPerfOfHloModule(const HloModule* hlo_module);
    std::vector<double> AnalyticalMemoryCostOfHloModule(const HloModule* fw_module, const HloModule* bw_module, 
                                                        const HloModule* apply_grad_module);

    class MemoryOffloader {
        public:
            MemoryOffloader(const HloModule* fw_module, const HloModule* bw_module, 
                            const HloModule* apply_grad_module, bool force_use_fp16)
                : force_use_fp16_(force_use_fp16), fw_module_(fw_module), 
                  bw_module_(bw_module), apply_grad_module_(apply_grad_module) {}

            // The entrance for memory offload cost estimation
            void EstimateOffloadCost();

            // Return the large param & params size
            std::pair<double, double> ParameterAnalysis(const HloModule* module) const;

            // Helper func to get the hlo module alloc memory
            double GetHloMoudleAllocMemory(const HloModule* module);

            // Offload strategy 3            
            void OffloadViaStrategy3();
            // Binary subgraph partition
            int64_t LowerBoundSubGraph(const HloModule* fw_module, int64_t slice_start, int64_t slice_end);
            // Lineary parition subgraph
            // Collect partition candidates
            void CollectPartitionCandidates();
            
            // Show the memory infos during offload process
            void LoggingMemoryInfos(int type) const;

            // Set hardware Configs
            void SetHardwareConfigs(double on_chip_mem, double mem_bandwidth) {
                on_chip_memory_ = on_chip_mem;
                memory_bandwidth_ = mem_bandwidth;
            }

            // Results getter
            std::vector<double> GetStagesCost() const { 
                return {fw_offload_cost_, bw_offload_cost_, apply_grad_offload_cost_};
            }
        
        private:
            // General members
            bool force_use_fp16_;

            // HloModules
            const HloModule* fw_module_; 
            const HloModule* bw_module_; 
            const HloModule* apply_grad_module_;
            
            // HloModules alloc memory infos
            double fw_alloc_memory_ = 0;
            double bw_alloc_memory_ = 0; 
            double apply_grad_alloc_memory_ = 0;

            // params & grads infos
            // note sizeof(grads) == sizeof(params)
            double params_size_ = 0;
            double grads_size_ = 0;
            double max_grad_size_ = 0;
            
            // offload cost for each stage
            double fw_offload_cost_ = 0;
            double bw_offload_cost_ = 0;
            double apply_grad_offload_cost_ = 0;

            // diff strategies memory 
            double total_alloc_mem_ = 0;
            double strategy_1_alloc_mem_ = 0;
            double strategy_2_alloc_mem_ = 0;

            // Offload strategy 3
            std::vector<int64_t> partition_candidates_;
            std::vector<int64_t> slice_pos_;
            std::vector<double> in_out_params_; // data transfer in subgroups
            std::vector<double> slice_mems_;
            double debug_minimal_mem_ = 1e20;   // only use for debug
            std::pair<int64_t, int64_t> minimal_mem_range_;

            // Hardware related configs
            double on_chip_memory_ = 0;
            double memory_bandwidth_ = 0;
    };

}  // namespace analytical_perf
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ANALYTICAL_PERF_MODEL_H_
