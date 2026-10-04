#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ncclComm_t g_comm = nullptr;
int g_rank = -1;
int g_world_size = -1;
std::mutex g_mutex;

ncclRedOp_t redop_from_int(int op) {
    switch (op) {
        case 0:
            return ncclSum;
        case 1:
            return ncclProd;
        case 2:
            return ncclMax;
        case 3:
            return ncclMin;
        case 4:
            return ncclAvg;
        default:
            throw std::runtime_error("nccl: redop must be 0=sum,1=prod,2=max,3=min,4=avg");
    }
}

ncclDataType_t dtype_from_torch(torch::ScalarType t) {
    switch (t) {
        case torch::kFloat32:
            return ncclFloat32;
        case torch::kFloat16:
            return ncclFloat16;
        case torch::kBFloat16:
            return ncclBfloat16;
        case torch::kInt32:
            return ncclInt32;
        case torch::kInt64:
            return ncclInt64;
        case torch::kUInt8:
            return ncclUint8;
        default:
            throw std::runtime_error("nccl: unsupported dtype");
    }
}

void check_comm() {
    if (g_comm == nullptr) {
        throw std::runtime_error("nccl: call init(rank, world_size, unique_id) first");
    }
}

size_t numel(const torch::Tensor& t) {
    return static_cast<size_t>(t.numel());
}

cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, ": CUDA tensor required");
    TORCH_CHECK(t.is_contiguous(), name, ": contiguous tensor required");
}

}  // namespace

std::string get_unique_id() {
    ncclUniqueId id;
    ncclResult_t st = ncclGetUniqueId(&id);
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclGetUniqueId: ") + ncclGetErrorString(st));
    }
    return std::string(reinterpret_cast<const char*>(&id), sizeof(ncclUniqueId));
}

void init_comm(int rank, int world_size, const std::string& unique_id_bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_comm != nullptr) {
        throw std::runtime_error("nccl: already initialized");
    }
    if (unique_id_bytes.size() != sizeof(ncclUniqueId)) {
        throw std::runtime_error("nccl: unique_id must be exactly sizeof(ncclUniqueId) bytes");
    }
    ncclUniqueId id;
    std::memcpy(&id, unique_id_bytes.data(), sizeof(ncclUniqueId));
    ncclResult_t st = ncclCommInitRank(&g_comm, world_size, id, rank);
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclCommInitRank: ") + ncclGetErrorString(st));
    }
    g_rank = rank;
    g_world_size = world_size;
}

int rank() {
    return g_rank;
}

int world_size() {
    return g_world_size;
}

void destroy_comm() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_comm != nullptr) {
        ncclCommDestroy(g_comm);
        g_comm = nullptr;
    }
    g_rank = -1;
    g_world_size = -1;
}

void all_reduce(torch::Tensor tensor, int redop, bool inplace) {
    check_comm();
    check_cuda_tensor(tensor, "all_reduce");
    const auto dt = dtype_from_torch(tensor.scalar_type());
    const size_t count = numel(tensor);
    void* send = tensor.data_ptr();
    void* recv = inplace ? send : tensor.data_ptr();
    if (!inplace) {
        throw std::runtime_error("nccl: out-of-place all_reduce not exposed; pass a separate output tensor to all_reduce_out");
    }
    ncclResult_t st =
        ncclAllReduce(send, recv, count, dt, redop_from_int(redop), g_comm, current_stream());
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce: ") + ncclGetErrorString(st));
    }
}

void all_reduce_out(torch::Tensor input, torch::Tensor output, int redop) {
    check_comm();
    check_cuda_tensor(input, "all_reduce_out input");
    check_cuda_tensor(output, "all_reduce_out output");
    TORCH_CHECK(input.sizes() == output.sizes(), "all_reduce_out: shape mismatch");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "all_reduce_out: dtype mismatch");
    const auto dt = dtype_from_torch(input.scalar_type());
    const size_t count = numel(input);
    ncclResult_t st = ncclAllReduce(input.data_ptr(), output.data_ptr(), count, dt, redop_from_int(redop),
                                    g_comm, current_stream());
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce: ") + ncclGetErrorString(st));
    }
}

void broadcast(torch::Tensor tensor, int root) {
    check_comm();
    check_cuda_tensor(tensor, "broadcast");
    const auto dt = dtype_from_torch(tensor.scalar_type());
    ncclResult_t st = ncclBroadcast(tensor.data_ptr(), tensor.data_ptr(), numel(tensor), dt, root, g_comm,
                                    current_stream());
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclBroadcast: ") + ncclGetErrorString(st));
    }
}

void all_gather(torch::Tensor input, torch::Tensor output) {
    check_comm();
    check_cuda_tensor(input, "all_gather input");
    check_cuda_tensor(output, "all_gather output");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "all_gather: dtype mismatch");
    TORCH_CHECK(static_cast<int64_t>(numel(output)) == numel(input) * g_world_size,
                "all_gather: output.numel() must be input.numel() * world_size");
    const auto dt = dtype_from_torch(input.scalar_type());
    ncclResult_t st =
        ncclAllGather(input.data_ptr(), output.data_ptr(), numel(input), dt, g_comm, current_stream());
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllGather: ") + ncclGetErrorString(st));
    }
}

void reduce_scatter(torch::Tensor input, torch::Tensor output) {
    check_comm();
    check_cuda_tensor(input, "reduce_scatter input");
    check_cuda_tensor(output, "reduce_scatter output");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "reduce_scatter: dtype mismatch");
    TORCH_CHECK(static_cast<int64_t>(numel(input)) == numel(output) * g_world_size,
                "reduce_scatter: input.numel() must be output.numel() * world_size");
    const auto dt = dtype_from_torch(input.scalar_type());
    ncclResult_t st = ncclReduceScatter(input.data_ptr(), output.data_ptr(), numel(output), dt, ncclSum, g_comm,
                                        current_stream());
    if (st != ncclSuccess) {
        throw std::runtime_error(std::string("ncclReduceScatter: ") + ncclGetErrorString(st));
    }
}

void group_all_reduce(torch::Tensor a, torch::Tensor b, int redop) {
    check_comm();
    check_cuda_tensor(a, "group_all_reduce a");
    check_cuda_tensor(b, "group_all_reduce b");
    const auto dt_a = dtype_from_torch(a.scalar_type());
    const auto dt_b = dtype_from_torch(b.scalar_type());
    ncclRedOp_t op = redop_from_int(redop);
    ncclGroupStart();
    ncclResult_t st_a = ncclAllReduce(a.data_ptr(), a.data_ptr(), numel(a), dt_a, op, g_comm, current_stream());
    ncclResult_t st_b = ncclAllReduce(b.data_ptr(), b.data_ptr(), numel(b), dt_b, op, g_comm, current_stream());
    ncclGroupEnd();
    if (st_a != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce (group a): ") + ncclGetErrorString(st_a));
    }
    if (st_b != ncclSuccess) {
        throw std::runtime_error(std::string("ncclAllReduce (group b): ") + ncclGetErrorString(st_b));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_unique_id", &get_unique_id, "NCCL unique id blob (rank 0 generates, broadcast to peers)");
    m.def("init", &init_comm, "ncclCommInitRank");
    m.def("destroy", &destroy_comm, "ncclCommDestroy");
    m.def("rank", &rank, "local rank after init");
    m.def("world_size", &world_size, "world size after init");
    m.def("all_reduce", &all_reduce, "in-place ncclAllReduce (redop: 0=sum..4=avg)");
    m.def("all_reduce_out", &all_reduce_out, "ncclAllReduce into separate output");
    m.def("broadcast", &broadcast, "ncclBroadcast in-place");
    m.def("all_gather", &all_gather, "ncclAllGather");
    m.def("reduce_scatter", &reduce_scatter, "ncclReduceScatter sum");
    m.def("group_all_reduce", &group_all_reduce, "two in-place all_reduces inside ncclGroupStart/End");
}
