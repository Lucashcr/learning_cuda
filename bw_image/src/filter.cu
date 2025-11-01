#include <cstdio>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void filterKernel(unsigned char *d_img, int width, int height,
                             int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  if (channels == 1)
    return;

  unsigned long long idx = (y * width + x) * channels;

  unsigned char r = d_img[idx + 0];
  unsigned char g = d_img[idx + 1];
  unsigned char b = d_img[idx + 2];

  float gray_f = 0.299f * r + 0.587f * g + 0.114f * b;
  unsigned char gray = (unsigned char)(gray_f + 0.5f);

  d_img[idx + 0] = gray;
  d_img[idx + 1] = gray;
  d_img[idx + 2] = gray;
}

extern "C" void applyBlackAndWhiteFilter(unsigned char *h_img, const int width,
                                         const int height, const int channels) {
  printf("running black and white filter...\n");

  unsigned long long size = (unsigned long long)width * height * channels;
  size_t bytes = size * sizeof(unsigned char);

  unsigned char *d_img = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img, bytes));
  CUDA_CHECK(cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  filterKernel<<<grid, block>>>(d_img, width, height, channels);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_img));

  printf("finish black and white filter!\n");
}
