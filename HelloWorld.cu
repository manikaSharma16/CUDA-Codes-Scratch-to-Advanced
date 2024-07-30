/*Tutorial followed: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
*/

__global__ void helloWorld(){ // __global__ specifies that this kernel(function) is to be run on device(GPU)
    printf("Hello World from GPU!\n");
}

// host code(CPU code)
int main() {
    helloWorld<<<1,1>>>(); // launch this kernel on just 1 thread of a block
    return 0;
}
