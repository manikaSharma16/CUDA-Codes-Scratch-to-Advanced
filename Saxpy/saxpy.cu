#include <iostream>
#include <cuda_runtime.h> // For CUDA functions and types

// __global__ means this is a kernel and will run on GPU
__global__ void saxpy(int n, float a, float *x, float *y) {

    // we parallelized the for loop
    int i = blockIdx.x * blockDim.x + threadIdx.x; // built in values for every single thread. Every thread knows where it is in a grid, then block. Compute global thread id
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 1000; // Example size of the vectors

     // Following the style used in many NVIDIA CUDA examples, we use the prefix h_ in naming pointer variables for memory allocated in CPU memory and d_ for pointers for memory allocated in GPU memory.
    float *h_x, *h_y; // Host memory
    float *d_x, *d_y; // Device memory

    // allocate host memory for h_x and h_y and initialize contents
    // cudaMalloc: invokes the GPU driver and asks it to allocate memory on the GPU for use by the program. 
    h_x = (float*) malloc(n * sizeof(float));
    h_y = (float*) malloc(n * sizeof(float));
    
    if (h_x == nullptr || h_y == nullptr) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize host memory
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_x, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_y, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x);
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from host to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x);
        cudaFree(d_y);
        return 1;
    }
    err = cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from host to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x);
        cudaFree(d_y);
        return 1;
    }

    int threadsPerBlock = 256; 
    int nblocks = (n + threadsPerBlock - 1) / threadsPerBlock; // Padding if the threads in block is less than desired(take round off for the block)
    saxpy<<<nblocks, threadsPerBlock>>>(n, 2.0f, d_x, d_y);

    // Check for any errors launching the kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x);
        cudaFree(d_y);
        return 1;
    }

    // Copy result from device to host
    err = cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from device to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_x);
        cudaFree(d_y);
        return 1;
    }

    // Print the result for verification, comment if doing time analysis
    for (int i = 0; i < n; ++i) {
        std::cout << "h_y[" << i << "] = " << h_y[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(h_x);
    free(h_y);

    return 0;
}
