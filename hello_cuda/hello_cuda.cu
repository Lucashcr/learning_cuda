#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello World from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    std::cout << "Hello World from CPU!" << std::endl;

    helloFromGPU<<<1, 10>>>();

    // Verifica se houve erro no lançamento
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Erro no kernel: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();

    return 0;
}

