#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <cstring>
#include <stdexcept>
#include <string>

//pointer to comm. Filled by ncclCommInitRank
static ncclComm_t g_comm = nullptr;

std::string get_unique_id() {
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) {
        throw std::runtime_error("ncclGetUniqueId failed");
    }
    return std::string(reinterpret_cast<char*>(&id), sizeof(id));
}

void init_comm(int rank, int world_size, const std::string& id_bytes) {
    if (g_comm) { 
        throw std::runtime_error("nccl already initialized");
    }
    ncclUniqueId id;
    std::memcpy(&id, id_bytes.data(), sizeof(id));
    if (ncclCommInitRank(&g_comm, world_size, id, rank) != ncclSuccess) { //world_size = nranks
        throw std::runtime_error("ncclCommInitRank failed");
    }
}


void destroy_comm() {
    if (g_comm) {
        ncclCommDestroy(g_comm);
        g_comm = nullptr;
    }
}


void all_reduce_sum(torch::Tensor t) {
    if (!g_comm) {
        throw std::runtime_error("call init first");
    }   
    auto* p = t.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (ncclAllReduce(p, p, t.numel(), ncclFloat32, ncclSum, g_comm, stream) != ncclSuccess) {
        throw std::runtime_error("ncclAllReduce failed");
    }
}

void all_gather(torch::Tensor in, torch::Tensor out) {
    if (!g_comm) {
        throw std::runtime_error("call init first");
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (ncclAllGather(in.data_ptr<float>(), out.data_ptr<float>(), in.numel(), ncclFloat32, g_comm,
                      stream) != ncclSuccess) {
        throw std::runtime_error("ncclAllGather failed");
    }
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_unique_id", &get_unique_id);
    m.def("init", &init_comm);
    m.def("destroy", &destroy_comm);
    m.def("all_reduce_sum", &all_reduce_sum);
    m.def("all_gather", &all_gather);
}
