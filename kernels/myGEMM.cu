#include <cuda_runtime.h>

#define tilesize 32
#define TM 4 //row granularity of a thread
#define TN 2
//I add the parameter TN to also expand out more elements of B. TM loads 4 rows of A. TN will load 2 columns of B

__device__ float GELU(float x) {
    float k = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}


__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta, bool GELU_bool) {
    int col = (blockDim.x * blockIdx.x + threadIdx.x)*TN;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;

    __shared__ float Atile[tilesize*TM][tilesize], Btile[tilesize][tilesize*TN]; 

    //the "f" sums represent the first row and the "s" sums represent the second row
    float sum1f = 0.0f, sum2f = 0.0f, sum3f = 0.0f, sum4f = 0.0f, sum1s = 0.0f, sum2s = 0.0f, sum3s = 0.0f, sum4s = 0.0f;

    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + tilesize - 1)/tilesize; tile++) {
        int tilecolA = tile * tilesize + threadIdx.x;
        int tilerowB = tile * tilesize + threadIdx.y;
        //load in 4 rows at a time into Atile. Can introduce a loop to make this dynamic to TM
        Atile[threadIdx.y * TM][threadIdx.x] = (row < M && tilecolA < K) ? A[row * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 1][threadIdx.x] = (row + 1 < M && tilecolA < K) ? A[(row + 1) * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 2][threadIdx.x] = (row + 2 < M && tilecolA < K) ? A[(row + 2) * K + tilecolA] : 0.0f;
        Atile[threadIdx.y * TM + 3][threadIdx.x] = (row + 3 < M && tilecolA < K) ? A[(row + 3) * K + tilecolA] : 0.0f;

        //load in 2 columns at a time of B tile
        Btile[threadIdx.y][threadIdx.x * TN] = (tilerowB < K && col < N) ? B[tilerowB * N + col] : 0.0f;
        Btile[threadIdx.y][threadIdx.x * TN + 1] = (tilerowB < K && col + 1 < N) ? B[tilerowB * N + col + 1] : 0.0f;
        __syncthreads();

        for (int p = 0; p < tilesize; p++) {
            float B_reuse_f = Btile[p][threadIdx.x * TN];
            float B_reuse_s = Btile[p][threadIdx.x * TN + 1];
            sum1f += Atile[threadIdx.y * TM][p] * B_reuse_f;
            sum2f += Atile[threadIdx.y * TM + 1][p] * B_reuse_f;
            sum3f += Atile[threadIdx.y * TM + 2][p] * B_reuse_f;
            sum4f += Atile[threadIdx.y * TM + 3][p] * B_reuse_f;

            sum1s += Atile[threadIdx.y * TM][p] * B_reuse_s;
            sum2s += Atile[threadIdx.y * TM + 1][p] * B_reuse_s;
            sum3s += Atile[threadIdx.y * TM + 2][p] * B_reuse_s;
            sum4s += Atile[threadIdx.y * TM + 3][p] * B_reuse_s;

        }
        
        __syncthreads();
    }

    //I think for my learning purposes it's good for me to map all this out. 
    if (col < N) {
        if (row < M) {
            float x = alpha * sum1f + beta * C[row * N + col];
            C[row * N + col] = GELU_bool ? GELU(x) : x;
            if (col + 1 < N) {
                float x = alpha * sum1s + beta * C[row * N + col + 1];
                C[row * N + col + 1] = GELU_bool ? GELU(x) : x;
            }
        }
        if (row + 1 < M) {
            float x = alpha * sum2f + beta * C[(row+1) * N + col];
            C[(row+1) * N + col] = GELU_bool ? GELU(x) : x;
            if (col + 1 < N) {
                float x = alpha * sum2s + beta * C[(row+1) * N + col + 1];
                C[(row+1) * N + col + 1] = GELU_bool ? GELU(x) : x;
            }
        }
        if (row + 2 < M) {
            float x = alpha * sum3f + beta * C[(row+2) * N + col];
            C[(row+2) * N + col] = GELU_bool ? GELU(x) : x;
            if (col + 1 < N) {
                float x = alpha * sum3s + beta * C[(row+2) * N + col + 1];
                C[(row+2) * N + col + 1] = GELU_bool ? GELU(x) : x;
            }
        }
        if (row + 3 < M) {
            float x = alpha * sum4f + beta * C[(row+3) * N + col];
            C[(row+3) * N + col] = GELU_bool ? GELU(x) : x;
            if (col + 1 < N) {
                float x = alpha * sum4s + beta * C[(row+3) * N + col + 1];
                C[(row+3) * N + col + 1] = GELU_bool ? GELU(x) : x;
            }
        }
    }
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta, bool GELU_bool=false) {
    dim3 threads(tilesize, tilesize);
    dim3 blocks((N + threads.x * TN - 1) / (threads.x * TN), 
                (M + threads.y * TM - 1) / (threads.y * TM));
    
    //for the sake of my QKV projections and transformer MLPs, I am not using biases so beta will be set to 0.
    if (beta == 0.0f) cudaMemset(C, 0, M * N * sizeof(float)); 
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta, GELU_bool);
    return cudaGetLastError();
}

