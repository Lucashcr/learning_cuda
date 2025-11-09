#include <iostream>

#define MAT_SIZE 3

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void multiplyKernel(int *d_A, int *d_B, int *d_C) {
  int matSize = blockDim.x;

  int ci = blockIdx.x;
  int cj = threadIdx.x;

  for (int step = 0; step < matSize; step++) {
    int aIdx = ci * matSize + step;
    int bIdx = step * matSize + cj;
    int cIdx = ci * matSize + cj;
    d_C[cIdx] += d_A[aIdx] * d_B[bIdx];
  }
}

int main(int argc, char **argv) {
  int h_A[MAT_SIZE][MAT_SIZE]{1, 2, 3, 4, 5, 6, 7, 8, 9};
  int h_B[MAT_SIZE][MAT_SIZE]{1, 2, 3, 4, 5, 6, 7, 8, 9};
  int h_C[MAT_SIZE][MAT_SIZE];

  std::cout << "A Matrix" << std::endl;
  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      std::cout << h_A[i][j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "B Matrix" << std::endl;
  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      std::cout << h_B[i][j] << " ";
    }
    std::cout << std::endl;
  }

  size_t bytes = MAT_SIZE * MAT_SIZE * sizeof(int);
  int *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemset(d_C, 0, bytes));

  dim3 block(MAT_SIZE);
  dim3 grid(MAT_SIZE);
  multiplyKernel<<<grid, block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  std::cout << "C Matrix" << std::endl;
  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      std::cout << h_C[i][j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
