#include <iostream>

void HANDLE_ERROR(cudaError_t error);

__global__ void kernel_basic_1(int a, int b, int *c) {
    *c = a + b;
}

void basic_1() {
    int c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(int)));
    kernel_basic_1<<<1,1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);
}

void basic_2() {
    cudaDeviceProp prop{};
    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("   --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap) {
            printf("Enabled\n");
        } else {
            printf("Disabled\n");
        }
        printf("   --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture alignment: %ld\n", prop.textureAlignment);
        printf("   --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads dimensions: (%d, %d. %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}

__global__ void kernel_basic_2(const int *a, const int *b, int *c, int N) {
    int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void basic_3() {
    int N = 10;
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_c, N * sizeof(int)));
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    kernel_basic_2<<<N,1>>>(dev_a, dev_b, dev_c, N);
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

int main() {
    return 0;
}

void HANDLE_ERROR(cudaError_t error) {

}


