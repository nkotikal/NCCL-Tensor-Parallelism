#include <cuda_runtime.h>

#define tilesize 32
#define TM 4 //row granularity of a thread
//we implement 4-element 1D register tiling in order to compute 4 elements per thread. 


//1D register tiled handwritten GEMM
__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;

    __shared__ float Atile[tilesize*TM][tilesize], Btile[tilesize][tilesize]; //allocate 128 rows for A to be able to compute 4 per thread


    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + tilesize - 1)/tilesize; tile++) {
        int tilecolA = tile * tilesize + threadIdx.x;
        int tilerowB = tile * tilesize + threadIdx.y;
        //load in 4 rows at a time into Atile. Can introduce a loop to make this dynamic to TM
        Atile[threadIdx.y * TM][threadIdx.x] = (row < M && tilecolA < K) ? A[row * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 1][threadIdx.x] = (row + 1 < M && tilecolA < K) ? A[(row + 1) * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 2][threadIdx.x] = (row + 2 < M && tilecolA < K) ? A[(row + 2) * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 3][threadIdx.x] = (row + 3 < M && tilecolA < K) ? A[(row + 3) * K + tilecolA] : 0.0f;

        Btile[threadIdx.y][threadIdx.x] = (tilerowB < K && col < N) ? B[tilerowB * N + col] : 0.0f;
        __syncthreads();

        for (int p = 0; p < tilesize; p++) {
            float B_reuse = Btile[p][threadIdx.x]; //not REALLY necessary because compiler will do this anyway, but wanted to show what the point was of this
            sum1 += Atile[threadIdx.y * TM][p] * B_reuse;
            sum2 += Atile[threadIdx.y * TM + 1][p] * B_reuse;
            sum3 += Atile[threadIdx.y * TM + 2][p] * B_reuse;
            sum4 += Atile[threadIdx.y * TM + 3][p] * B_reuse;

        }
        
        __syncthreads();
    }

    if (col < N) {
        if (row < M)
            C[row * N + col] = alpha * sum1 + beta * C[row * N + col];
        if (row + 1 < M)
            C[(row+1) * N + col] = alpha * sum2 + beta * C[(row+1) * N + col];
        if (row + 2 < M)
            C[(row+2) * N + col] = alpha * sum3 + beta * C[(row+2) * N + col];
        if (row + 3 < M)
            C[(row+3) * N + col] = alpha * sum4 + beta * C[(row+3) * N + col];
    }

}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(tilesize, tilesize);
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (M + threads.y * TM - 1) / (threads.y * TM));
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

