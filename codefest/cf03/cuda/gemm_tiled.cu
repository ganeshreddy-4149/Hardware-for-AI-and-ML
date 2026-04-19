/*
 * gemm_tiled.cu
 * Tiled (shared-memory) CUDA kernel for 1024x1024 FP32 matrix multiply
 * Uses TILE x TILE shared memory blocks to reduce DRAM traffic
 * Each tile element is reused TILE times instead of being loaded fresh
 * ECE 410/510 - Codefest 3, CLLM Task 1b
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
#define N    1024   // Matrix dimension (N x N)
#define TILE 8      // Tile size as required by assignment (T=8)
                    // Each block = TILE x TILE = 64 threads
                    // Each thread loads 1 element of A tile + 1 of B tile

// ─────────────────────────────────────────────
// CUDA ERROR CHECKING MACRO
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
// TILED KERNEL
// Each thread block handles a TILE x TILE output tile of C
// Threads cooperatively load tiles of A and B into shared memory
// Then compute partial dot products using the fast shared memory
// ─────────────────────────────────────────────
__global__ void gemm_tiled(const float* A, const float* B, float* C, int n)
{
    // ── Shared memory tiles ───────────────────────────────────────────────
    // Each block has its own private copy of these tiles in on-chip SRAM
    // All TILE*TILE threads in this block can read/write these arrays
    // Size: TILE*TILE*4 bytes = 8*8*4 = 256 bytes per tile = 512 bytes total
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    // ── Thread's local position within the tile ───────────────────────────
    int tx = threadIdx.x;   // local column (0 to TILE-1)
    int ty = threadIdx.y;   // local row    (0 to TILE-1)

    // ── Thread's global position in the output matrix C ───────────────────
    // This is the element of C that THIS thread is responsible for computing
    int col = blockIdx.x * TILE + tx;   // global column in C
    int row = blockIdx.y * TILE + ty;   // global row in C

    // ── Accumulator for partial dot product ───────────────────────────────
    // This stays in a register (fastest memory) across all tile steps
    float sum = 0.0f;

    // ── Number of tiles we sweep through in the K dimension ──────────────
    int num_tiles = (n + TILE - 1) / TILE;   // = 1024/8 = 128 tile steps

    // ── Main tiling loop — sweep across K dimension one tile at a time ────
    for (int t = 0; t < num_tiles; t++) {

        // ── STEP 1: LOAD tile of A into shared memory ─────────────────────
        // Each thread loads ONE element: A[row][t*TILE + tx]
        // row    = this thread's global row
        // t*TILE = start of current tile in K dimension
        // tx     = this thread's column offset within the tile
        int a_col = t * TILE + tx;   // column index into A
        if (row < n && a_col < n)
            tile_A[ty][tx] = A[row * n + a_col];
        else
            tile_A[ty][tx] = 0.0f;   // padding for edge cases

        // ── STEP 1b: LOAD tile of B into shared memory ────────────────────
        // Each thread loads ONE element: B[t*TILE + ty][col]
        // t*TILE + ty = row index into B (which row of B tile to load)
        // col         = this thread's global column
        int b_row = t * TILE + ty;   // row index into B
        if (b_row < n && col < n)
            tile_B[ty][tx] = B[b_row * n + col];
        else
            tile_B[ty][tx] = 0.0f;   // padding for edge cases

        // ── STEP 2: SYNCHRONIZE ───────────────────────────────────────────
        // CRITICAL: ALL threads in the block must finish loading their
        // element into shared memory before ANY thread starts computing.
        // Without this, threads could read uninitialized shared memory!
        __syncthreads();

        // ── STEP 3: COMPUTE partial dot product from shared memory ─────────
        // This inner loop uses ONLY shared memory (on-chip SRAM, ~1 TB/s)
        // NOT DRAM. This is where the speedup comes from!
        // Each of the TILE=8 elements is reused by multiple threads
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
            //     ^A[row][k]       ^B[k][col]
            //     both from fast shared memory, NOT DRAM!
        }

        // ── STEP 4: SYNCHRONIZE again before next tile load ───────────────
        // Ensure ALL threads finish computing from current tiles before
        // any thread overwrites shared memory with the NEXT tile's data
        __syncthreads();
    }

    // ── Write final result to global memory (DRAM) ────────────────────────
    // Only ONE write to DRAM per thread (vs reading N elements from DRAM in naive)
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// ─────────────────────────────────────────────
// CPU REFERENCE: verify GPU result is correct
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
    printf("  Tiled CUDA GEMM — %dx%d FP32, Tile Size = %d\n", N, N, TILE);
    printf("==========================================================\n\n");

    // ── Memory size ──────────────────────────
    size_t bytes = (size_t)N * N * sizeof(float);
    printf("[1] Matrix size  : %d x %d\n", N, N);
    printf("[1] Tile size    : %d x %d\n", TILE, TILE);
    printf("[1] Memory/matrix: %.2f MB\n", bytes / 1e6);
    printf("[1] Tiles per dim: %d (= N/T = %d/%d)\n\n",
           N / TILE, N, TILE);

    // ── Allocate HOST memory ─────────────────
    float* h_A    = (float*)malloc(bytes);
    float* h_B    = (float*)malloc(bytes);
    float* h_C    = (float*)malloc(bytes);
    float* h_Cref = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_Cref) {
        fprintf(stderr, "Host malloc failed!\n");
        return EXIT_FAILURE;
    }

    // ── Initialize with same seed as naive for fair comparison ──
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }
    printf("[2] Matrices initialized (same seed as naive for comparison)\n\n");

    // ── Allocate DEVICE memory ───────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    printf("[3] GPU memory allocated: %.2f MB total\n\n", 3.0 * bytes / 1e6);

    // ── Copy CPU → GPU ───────────────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    printf("[4] Data copied from CPU to GPU\n\n");

    // ── Thread block = TILE x TILE = 8x8 = 64 threads ──
    // Each block computes one TILE x TILE output tile of C
    dim3 block(TILE, TILE);

    // ── Grid = (N/TILE) x (N/TILE) = 128x128 = 16384 blocks ──
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    printf("[5] Launch config:\n");
    printf("    Tile size  : %d x %d = %d threads/block\n",
           TILE, TILE, TILE * TILE);
    printf("    Grid size  : %d x %d = %d blocks\n",
           grid.x, grid.y, grid.x * grid.y);
    printf("    Total threads: %d\n",
           grid.x * grid.y * TILE * TILE);
    printf("    Tile steps (K): %d\n\n", N / TILE);

    // ── Warm-up run ──────────────────────────
    printf("[6] Running warm-up kernel...\n");
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("    Warm-up done\n\n");

    // ── Timed run ────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    printf("[7] Kernel execution time: %.4f ms\n\n", time_ms);

    // ── Compute GFLOP/s ──────────────────────
    double flops          = 2.0 * (double)N * N * N;
    double gflops         = flops / 1e9;
    double time_sec       = time_ms / 1000.0;
    double gflops_per_sec = gflops / time_sec;

    printf("[8] Performance:\n");
    printf("    Total FLOPs     : %.3e\n", flops);
    printf("    Time            : %.4f ms = %.6f s\n", time_ms, time_sec);
    printf("    Achieved GFLOP/s: %.2f\n\n", gflops_per_sec);

    // ── Copy result back ─────────────────────
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("[9] Result copied back from GPU to CPU\n\n");

    // ── Correctness check ────────────────────
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

    // ── Shared memory info ───────────────────
    printf("[11] Shared memory used per block:\n");
    printf("     tile_A: %d x %d x 4 bytes = %d bytes\n",
           TILE, TILE, TILE * TILE * 4);
    printf("     tile_B: %d x %d x 4 bytes = %d bytes\n",
           TILE, TILE, TILE * TILE * 4);
    printf("     Total : %d bytes per block\n\n",
           2 * TILE * TILE * 4);

    // ── Summary ──────────────────────────────
    printf("==========================================================\n");
    printf("  SUMMARY — Tiled GEMM (Tile = %d)\n", TILE);
    printf("==========================================================\n");
    printf("  Matrix size        : %d x %d (FP32)\n", N, N);
    printf("  Tile size          : %d x %d\n", TILE, TILE);
    printf("  Kernel time        : %.4f ms\n", time_ms);
    printf("  Achieved GFLOP/s   : %.2f\n", gflops_per_sec);
    printf("  Max error vs CPU   : %.6f\n", max_error);
    printf("  Shared mem/block   : %d bytes\n", 2 * TILE * TILE * 4);
    printf("==========================================================\n");

    // ── Free memory ──────────────────────────
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
