#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel for matrix multiplication using tiling with dynamic shared memory
__global__ void matrixMulTiledDynamic(float *A, float *B, float *C, int M, int N, int K, int tileSize) {
    // Dynamically allocated shared memory
    extern __shared__ float sharedMemory[];
    float *tileA = sharedMemory;                            // Shared memory for tile of A
    float *tileB = sharedMemory + tileSize * tileSize;      // Shared memory for tile of B

    // Calculate the row and column for the current thread
    int row = threadIdx.y + blockIdx.y * tileSize;          // Row index in C
    int col = threadIdx.x + blockIdx.x * tileSize;          // Column index in C

    float value = 0.0f; // Accumulate partial results for the element in C

    // Loop over tiles of A and B required for this block's output tile
    for (int t = 0; t < (K + tileSize - 1) / tileSize; t++) {
        // Load tile of A into shared memory
        if (row < M && t * tileSize + threadIdx.x < K) {
            tileA[threadIdx.y * tileSize + threadIdx.x] = A[row * K + t * tileSize + threadIdx.x];
        } else {
            tileA[threadIdx.y * tileSize + threadIdx.x] = 0.0f; // Handle boundary condition
        }

        // Load tile of B into shared memory
        if (col < N && t * tileSize + threadIdx.y < K) {
            tileB[threadIdx.y * tileSize + threadIdx.x] = B[(t * tileSize + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y * tileSize + threadIdx.x] = 0.0f; // Handle boundary condition
        }

        __syncthreads(); // Ensure all threads have loaded their tiles

        // Perform the computation for this tile
        for (int i = 0; i < tileSize; i++) {
            value += tileA[threadIdx.y * tileSize + i] * tileB[i * tileSize + threadIdx.x];
        }

        __syncthreads(); // Ensure all threads have completed computation before next tile
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Function to determine tile size dynamically based on hardware and matrix dimensions
int calculateDynamicTileSize(int M, int N, int K) {
    // Query GPU properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Query properties of GPU 0
    int maxSharedMemoryPerBlock = deviceProp.sharedMemPerBlock; // Shared memory size in bytes

    // Determine optimal tile size dynamically
    int elementSize = sizeof(float);                              // Size of each matrix element
    int maxTileElements = maxSharedMemoryPerBlock / (2 * elementSize); // Total elements that fit in shared memory
    int tileSize = sqrt(maxTileElements);                         // Use square tiles for simplicity
    tileSize = min(tileSize, 32);                                 // Limit tile size to warp size for efficiency
    tileSize = min(tileSize, max(M, max(N, K)));                  // Adjust for matrix dimensions

    return tileSize; // Return calculated tile size
}

// Function to infer dimensions and launch the kernel
void optimizeAndLaunchKernel(float *A, float *B, float *C, size_t totalElements_A, size_t totalElements_B, size_t totalElements_C, int K) {
    // Infer matrix dimensions based on metadata
    int M = totalElements_A / K;   // Rows in A
    int N = totalElements_B / K;   // Columns in B

    // Check for valid dimensions
    if (M * K != totalElements_A || K * N != totalElements_B) {
        std::cerr << "Error: Matrix dimensions do not match the provided metadata." << std::endl;
        return;
    }

    // Calculate tile size dynamically
    int tileSize = calculateDynamicTileSize(M, N, K);

    // Configure grid and block dimensions
    dim3 blockDim(tileSize, tileSize);                            // Threads per block
    dim3 gridDim((N + tileSize - 1) / tileSize,                   // Grid dimensions for C's rows
                 (M + tileSize - 1) / tileSize);                  // Grid dimensions for C's columns

    // Shared memory size required per block
    size_t sharedMemorySize = 2 * tileSize * tileSize * sizeof(float); // Shared memory for tiles of A and B

    // Launch the kernel
    matrixMulTiledDynamic<<<gridDim, blockDim, sharedMemorySize>>>(A, B, C, M, N, K, tileSize);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    // Assume matrices are provided dynamically during runtime
    size_t totalElements_A = 2048 * 1024; // Total elements in matrix A
    size_t totalElements_B = 1024 * 512;  // Total elements in matrix B
    size_t totalElements_C = 2048 * 512;  // Total elements in matrix C (output)
    int K = 1024;                         // Columns in A and rows in B (shared dimension)

    // Allocate memory for matrices on the host
    float *h_A = new float[totalElements_A];
    float *h_B = new float[totalElements_B];
    float *h_C = new float[totalElements_C];

    // Initialize matrices (example: random or sequential values)
    for (size_t i = 0; i < totalElements_A; i++) h_A[i] = 1.0f; // Example initialization
    for (size_t i = 0; i < totalElements_B; i++) h_B[i] = 1.0f; // Example initialization

    // Allocate memory for matrices on the GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, totalElements_A * sizeof(float));
    cudaMalloc(&d_B, totalElements_B * sizeof(float));
    cudaMalloc(&d_C, totalElements_C * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, totalElements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalElements_B * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the optimized kernel
    optimizeAndLaunchKernel(d_A, d_B, d_C, totalElements_A, totalElements_B, totalElements_C, K);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, totalElements_C * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (optional, for correctness testing)
    for (size_t i = 0; i < totalElements_C; i++) {
        if (h_C[i] != K) { // Example validation
            std::cerr << "Error at index " << i << ": Expected " << K << ", got " << h_C[i] << std::endl;
            break;
        }
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
