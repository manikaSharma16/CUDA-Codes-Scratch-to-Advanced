/**
  SAXPY is part of the well-known Basic Linear Algebra Software (BLAS) library.
  It is useful for implementing higher-level matrix operations such as Gaussian elimination.
  Single-precision scalar value A times vector value X plus vector value Y, known as SAXPY.
*/

#include <iostream>
#include <cstdlib>  // For malloc and free
#include <ctime>    // For time
#include <cstring>  // For memset

void saxpy_serial(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 1000;  // Example size of the vectors
    float *x, *y;

    // Allocate CPU memory for x and y
    x = (float*) malloc(n * sizeof(float));
    y = (float*) malloc(n * sizeof(float));
    
    if (x == nullptr || y == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize x and y with some values
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    // Invoke serial SAXPY kernel
    saxpy_serial(n, 2.0f, x, y); // Perform SAXPY operation with a = 2.0

    // Print the result for verification, comment this part for time analysis
    for (int i = 0; i < n; ++i) {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    // Free allocated memory
    free(x);
    free(y);

    return 0;
}
