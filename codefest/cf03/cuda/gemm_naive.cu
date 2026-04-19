/*
 * gemm_naive.cu
 * Naive CUDA kernel for 1024x1024 FP32 matrix multiply C = A x B
 * One thread computes one output element C[row][col]
 * ECE 410/510 - Codefest 3, CLLM Task 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Matrix size
// ─────────────────────────────────────────────
#define N 1024          // Matrix is N x N
#define BLOCK_SIZE 32   // Each thread block is 32x32 = 1024 threads

// ─────────────────────────────────────────────
// CUDA ERROR CHECKING MACRO
// Wraps every CUDA call to catch errors immediately
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ─────────────────────────────────────────────
// NAIVE KERNEL
// Each thread computes exactly ONE element of C
// Thread at (row, col) computes C[row][col] = dot(A[row][:], B[:][col])
// ─────────────────────────────────────────────
__global__ void gemm_naive(const float* A, const float* B, float* C, int n)
{
    // Compute this thread's global row and column in matrix C
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard: make sure we don't go out of matrix bounds
    if (row >= n || col >= n) return;

    // Accumulate the dot product of A's row and B's column
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        // A is stored row-major: A[row][k] = A[row * n + k]
        // B is stored row-major: B[k][col] = B[k  * n + col]
        sum += A[row * n + k] * B[k * n + col];
    }

    // Write result to C
    C[row * n + col] = sum;
}

// ─────────────────────────────────────────────
// CPU REFERENCE: verify GPU result is correct
// Computes a small portion to keep it fast
// ─────────────────────────────────────────────
void cpu_gemm_verify(const float* A, const float* B, float* C_ref, int n, int check_size)
{
    for (int i = 0; i < check_size; i++) {
        for (int j = 0; j < check_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C_ref[i * n + j] = sum;
        }
    }
}

// ─────────────────────────────────────────────
// MAIN FUNCTION
// ─────────────────────────────────────────────
int main()
{
    printf("==========================================================\n");
    printf("  Naive CUDA GEMM — %dx%d FP32 Matrix Multiply\n", N, N);
    printf("==========================================================\n\n");

    // ── Matrix size in bytes ──────────────────
    size_t bytes = (size_t)N * N * sizeof(float);
    printf("[1] Matrix size : %d x %d\n", N, N);
    printf("[1] Memory/matrix: %.2f MB\n\n", bytes / 1e6);

    // ── Allocate HOST memory (CPU side) ───────
    float* h_A    = (float*)malloc(bytes);
    float* h_B    = (float*)malloc(bytes);
    float* h_C    = (float*)malloc(bytes);   // GPU result copied back here
    float* h_Cref = (float*)malloc(bytes);   // CPU reference result

    if (!h_A || !h_B || !h_C || !h_Cref) {
        fprintf(stderr, "Host malloc failed!\n");
        return EXIT_FAILURE;
    }

    // ── Initialize A and B with small random values ──
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;   // values 0.0 to 0.9
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }
    printf("[2] Matrices initialized with random FP32 values\n\n");

    // ── Allocate DEVICE memory (GPU side) ────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    printf("[3] GPU memory allocated: 3 x %.2f MB = %.2f MB total\n\n",
           bytes / 1e6, 3.0 * bytes / 1e6);

    // ── Copy data from CPU → GPU ──────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    printf("[4] Data copied from CPU to GPU\n\n");

    // ── Define thread block and grid dimensions ──
    // Block: 32x32 = 1024 threads per block
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // Grid: ceil(N/32) x ceil(N/32) blocks
    // For N=1024, BLOCK_SIZE=32: grid = 32x32 blocks
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("[5] Launch config:\n");
    printf("    Block size : %d x %d = %d threads\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE);
    printf("    Grid size  : %d x %d = %d blocks\n",
           grid.x, grid.y, grid.x * grid.y);
    printf("    Total threads: %d\n\n",
           grid.x * grid.y * BLOCK_SIZE * BLOCK_SIZE);

    // ── Warm-up run (GPU needs a warm-up to reach peak clock speed) ──
    printf("[6] Running warm-up kernel...\n");
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("    Warm-up done\n\n");

    // ── Timed run using CUDA Events ──────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Launch the naive kernel
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for GPU to finish completely
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Get elapsed time in milliseconds
    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("[7] Kernel execution time: %.4f ms\n\n", time_ms);

    // ── Compute GFLOP/s ───────────────────────
    // Total FLOPs = 2 * N^3 (N multiplications + N additions per output,
    //               N^2 output elements)
    double flops      = 2.0 * (double)N * N * N;
    double gflops     = flops / 1e9;
    double time_sec   = time_ms / 1000.0;
    double gflops_per_sec = gflops / time_sec;

    printf("[8] Performance:\n");
    printf("    Total FLOPs    : %.3e\n", flops);
    printf("    Time           : %.4f ms = %.6f s\n", time_ms, time_sec);
    printf("    Achieved GFLOP/s: %.2f\n\n", gflops_per_sec);

    // ── Copy result back from GPU → CPU ──────
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("[9] Result copied back from GPU to CPU\n\n");

    // ── Verify correctness against CPU result ─
    // Only check a 32x32 corner (full CPU check would take too long)
    int check_size = 32;
    cpu_gemm_verify(h_A, h_B, h_Cref, N, check_size);

    float max_error = 0.0f;
    for (int i = 0; i < check_size; i++) {
        for (int j = 0; j < check_size; j++) {
            float diff = fabsf(h_C[i * N + j] - h_Cref[i * N + j]);
            if (diff > max_error) max_error = diff;
        }
    }
    printf("[10] Correctness check (first %dx%d elements):\n", check_size, check_size);
    printf("     Max absolute error vs CPU: %.6f\n", max_error);
    if (max_error < 1e-2f)
        printf("     ✓ PASSED — GPU result matches CPU reference\n\n");
    else
        printf("     ✗ FAILED — Large error detected!\n\n");

    // ── Print Summary ─────────────────────────
    printf("==========================================================\n");
    printf("  SUMMARY — Naive GEMM\n");
    printf("==========================================================\n");
    printf("  Matrix size        : %d x %d (FP32)\n", N, N);
    printf("  Kernel time        : %.4f ms\n", time_ms);
    printf("  Achieved GFLOP/s   : %.2f\n", gflops_per_sec);
    printf("  Max error vs CPU   : %.6f\n", max_error);
    printf("==========================================================\n");

    // ── Free all memory ───────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Cref);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
