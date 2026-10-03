#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                  int M, int N, int K, float alpha, float beta, bool GELU_bool);

torch::Tensor gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                   double alpha, double beta, bool gelu) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "gemm: CUDA tensors only");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "gemm: float32 only");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32 && C.scalar_type() == torch::kFloat32, "gemm: float32 only");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "gemm: contiguous only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "gemm: 2D tensors only");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "gemm: B must be (K, N)");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "gemm: C must be (M, N)");

    cudaError_t err = launchGEMM(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                                 M, N, K, static_cast<float>(alpha), static_cast<float>(beta), gelu);
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "C = alpha * A @ B + beta * C, optional GELU epilogue");
}
