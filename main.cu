#include <iostream>
#include <vector>

#include "Helper.cuh"

#define imin(a,b) (a < b ? a : b)

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

__global__ void kernel_basic_3(const int *a, const int *b, int *c, int N) {
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
    kernel_basic_3<<<N,1>>>(dev_a, dev_b, dev_c, N);
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void kernel_basic_4(const int *a, const int *b, int *c, int N) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

void basic_4() {
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
    kernel_basic_3<<<1,N>>>(dev_a, dev_b, dev_c, N);
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void kernel_basic_5(const int *a, const int *b, int *c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void basic_5() {
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
    kernel_basic_3<<<128, 128>>>(dev_a, dev_b, dev_c, N);
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void kernel_dot(float *a, float *b, float * c) {
    const int N = 33*1024;
    const int threadsPerBlock = 256;
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (i == 0) {
        c[blockIdx.x] = cache[0];
    }
}

void dot() {
    const int N = 33*1024;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    partial_c = (float *) malloc(blocksPerGrid * sizeof(float));
    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_partial_c, blocksPerGrid * sizeof(float)));
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    kernel_dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }
#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float) (N - 1)));
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(a);
    free(b);
    free(partial_c);
}


__global__ void kernel_structDynamicArray(struct dynamic_array_struct *t) {
    printf("GPU sees t[0].k = %d\n", t[0].k);
    printf("GPU sees t[0].i[1] = %d\n", t[0].i[1]);
    printf("GPU sees t[0].n = %d\n", t[0].n);
    printf("GPU sees t[1].k = %d\n", t[1].k);
    printf("GPU sees t[1].i[1] = %d\n", t[1].i[1]);
    printf("GPU sees t[1].n = %d\n", t[1].n);
    t[0].k = 10;
    t[0].i[1] = 100;
    t[0].n = 1000;
    t[1].k = 20;
    t[1].i[1] = 200;
    t[1].n = 2000;
}

int struct_dynamic_array()
{
    struct dynamic_array_struct *th;
    th = (struct dynamic_array_struct *) malloc(2 * sizeof(struct dynamic_array_struct));
    th[0].i = (int *) malloc(sizeof(int) * 10);
    th[1].i = (int *) malloc(sizeof(int) * 10);
    th[0].k = 1;
    th[0].i[1] = 10;
    th[0].n = 100;
    th[1].k = 2;
    th[1].i[1] = 20;
    th[1].n = 200;

    auto *tt = (struct dynamic_array_struct *) malloc(2 * sizeof(dynamic_array_struct));
    memcpy(tt, th, 2 * sizeof(struct dynamic_array_struct));
    HANDLE_ERROR(cudaMalloc(&tt[0].i, sizeof(int) * 10));
    HANDLE_ERROR(cudaMalloc(&tt[1].i, sizeof(int) * 10));
    HANDLE_ERROR(cudaMemcpy(tt[0].i, th[0].i, 10 * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(tt[1].i, th[1].i, 10 * sizeof(int), cudaMemcpyHostToDevice));

    struct dynamic_array_struct *t_dev;
    HANDLE_ERROR(cudaMalloc(&t_dev, 2 * sizeof(struct dynamic_array_struct)));
    HANDLE_ERROR(cudaMemcpy(t_dev, tt, 2 * sizeof(struct dynamic_array_struct), cudaMemcpyHostToDevice));

    kernel_structDynamicArray<<< 1, 1 >>>(t_dev);   // Start GPU function
    HANDLE_ERROR(cudaDeviceSynchronize());   // Wait until GPU kernel function finishes !!

    HANDLE_ERROR(cudaMemcpy(tt, t_dev, 2 * sizeof(struct dynamic_array_struct), cudaMemcpyDeviceToHost));
    memcpy(th, tt, 2 * sizeof(struct dynamic_array_struct));
    th[0].i = (int *) malloc(sizeof(int) * 10);
    th[1].i = (int *) malloc(sizeof(int) * 10);
    HANDLE_ERROR(cudaMemcpy(th[0].i, tt[0].i, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(th[1].i, tt[1].i, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(tt[0].i));
    HANDLE_ERROR(cudaFree(tt[1].i));
    HANDLE_ERROR(cudaFree(t_dev));
    free(tt);

    printf("CPU sees th[0].k = %d\n", th[0].k);  // Now obtain the result !!
    printf("CPU sees th[0].i[1] = %d\n", th[0].i[1]);  // Now obtain the result !!
    printf("CPU sees th[0].n = %d\n", th[0].n);  // Now obtain the result !!
    printf("CPU sees th[1].k = %d\n", th[1].k);  // Now obtain the result !!
    printf("CPU sees th[1].i[1] = %d\n", th[1].i[1]);  // Now obtain the result !!
    printf("CPU sees th[1].n = %d\n", th[1].n);  // Now obtain the result !!

    free(th[0].i);
    free(th[1].i);
    free(th);

    return 0;
}

__global__ void kernel_vector(int *i) {
    printf("%d.\n", i[1]);
    i[1] = 100;
}

void vector() {
    std::vector<int> arr;
    for (int i = 0; i < 10; i++) arr.push_back(i);
    int *dev_arr;
    cudaMalloc((void **) &dev_arr, arr.size() * sizeof(int));
    cudaMemcpy(dev_arr, &arr[0], arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    kernel_vector<<<1,1>>>(dev_arr);
    HANDLE_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&arr[0], dev_arr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d.\n", arr[1]);
}

__global__ void kernel_structStaticArray2D(struct static_array2D_struct *t) {
    printf("GPU sees t[0].k = %d\n", t[0].k);
    printf("GPU sees t[0].i[1][1] = %d\n", t[0].i[1][1]);
    printf("GPU sees t[0].n = %d\n", t[0].n);
    printf("GPU sees t[1].k = %d\n", t[1].k);
    printf("GPU sees t[1].i[1][1] = %d\n", t[1].i[1][1]);
    printf("GPU sees t[1].n = %d\n", t[1].n);
    t[0].k = 10;
    t[0].i[1][1] = 100;
    t[0].n = 1000;
    t[1].k = 20;
    t[1].i[1][1] = 200;
    t[1].n = 2000;
}

void struct_static_array2D() {
    struct static_array2D_struct *th;
    th = (struct static_array2D_struct *) malloc(2 * sizeof(struct static_array2D_struct));
    th[0].k = 1;
    th[0].i[1][1] = 10;
    th[0].n = 100;
    th[1].k = 2;
    th[1].i[1][1] = 20;
    th[1].n = 200;

    struct static_array2D_struct *t_dev;
    HANDLE_ERROR(cudaMalloc(&t_dev, 2 * sizeof(struct static_array2D_struct)));
    HANDLE_ERROR(cudaMemcpy(t_dev, th, 2 * sizeof(struct static_array2D_struct), cudaMemcpyHostToDevice));

    kernel_structStaticArray2D<<<1,1>>>(t_dev);   // Start GPU function
    HANDLE_ERROR(cudaDeviceSynchronize());   // Wait until GPU kernel function finishes !!

    HANDLE_ERROR(cudaMemcpy(th, t_dev, 2 * sizeof(struct static_array2D_struct), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(t_dev));

    printf("CPU sees th[0].k = %d\n", th[0].k);  // Now obtain the result !!
    printf("CPU sees th[0].i[1][1] = %d\n", th[0].i[1][1]);  // Now obtain the result !!
    printf("CPU sees th[0].n = %d\n", th[0].n);  // Now obtain the result !!
    printf("CPU sees th[1].k = %d\n", th[1].k);  // Now obtain the result !!
    printf("CPU sees th[1].i[1][1] = %d\n", th[1].i[1][1]);  // Now obtain the result !!
    printf("CPU sees th[1].n = %d\n", th[1].n);  // Now obtain the result !!

    free(th);
}

__global__ void kernel_structStruct(struct struct_struct *t) {
    printf("GPU sees t[0].i.x= %d\n", t[0].i.x);
    printf("GPU sees t[1].i.y= %d\n", t[1].i.y);
    t[0].i.x = 10;
    t[1].i.y = 20;
}

void struct_struct() {
    struct struct_struct *th = (struct struct_struct *) malloc(2 * sizeof(struct struct_struct));
    th[0].i.x = 1;
    th[1].i.y = 2;
    struct struct_struct *t_dev;
    HANDLE_ERROR(cudaMalloc((void **) &t_dev, 2 * sizeof(struct struct_struct)));
    HANDLE_ERROR(cudaMemcpy(t_dev, th, 2 * sizeof(struct struct_struct), cudaMemcpyHostToDevice));
    kernel_structStruct<<<1,1>>>(t_dev);
    HANDLE_ERROR(cudaMemcpy(th, t_dev, 2 * sizeof(struct struct_struct), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(t_dev));
    printf("CPU sees th[0].i.x= %d\n", th[0].i.x);
    printf("CPU sees th[1].i.y= %d\n", th[1].i.y);
    free(th);
}

__global__ void kernel_pointerCheck(struct dynamic_array_struct *t) {
    printf("%d.\n", t[0].i[1]);
}

void pointercheck() {
    struct dynamic_array_struct *th;
    th = (struct dynamic_array_struct *) malloc(2 * sizeof(struct dynamic_array_struct));
    th[0].i = (int *) malloc(sizeof(int) * 10);
    th[1].i = (int *) malloc(sizeof(int) * 10);
    th[0].k = 1;
    th[0].i[1] = 10;
    th[0].n = 100;
    th[1].k = 2;
    th[1].i[1] = 20;
    th[1].n = 200;
    struct dynamic_array_struct *t_dev;
    HANDLE_ERROR(cudaMalloc(&t_dev, 2 * sizeof(struct dynamic_array_struct)));
    HANDLE_ERROR(cudaMemcpy(t_dev, th, 2 * sizeof(struct dynamic_array_struct), cudaMemcpyHostToDevice));
    kernel_pointerCheck<<<1,1>>>(t_dev);
    HANDLE_ERROR(cudaMemcpy(th, t_dev, 2 * sizeof(struct dynamic_array_struct), cudaMemcpyDeviceToHost));
    printf("%d.\n", th[0].i[1]);
    printf("%d.\n", th[1].i[1]);

}

__global__ void kernel_array2D(int **arr) {
    printf("GPU sees arr[1][1] = %d\n", arr[1][1]);
    arr[1][1] *= 10;
}

void array2D() {
    int **h_arr = (int **) malloc(10 * sizeof(int *));
    for (int i = 0; i < 10; i++) {
        h_arr[i] = (int *) malloc(10 * sizeof(int));
        for (int j = 0; j < 10; j++) {
            h_arr[i][j] = i + j;
        }
    }
    int **dev_arr;
    HANDLE_ERROR(cudaMalloc((void **) &dev_arr, 10 * sizeof(int *)));
    int *builder_arr[10];
    for (int i = 0; i < 10; i++) {
        HANDLE_ERROR(cudaMalloc((void **) &builder_arr[i], 10 * sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(builder_arr[i], h_arr[i], 10 * sizeof(int), cudaMemcpyHostToDevice));
    }
    HANDLE_ERROR(cudaMemcpy(dev_arr, builder_arr, 10 * sizeof(int *), cudaMemcpyHostToDevice));
    kernel_array2D<<<1,1>>>(dev_arr);
    HANDLE_ERROR(cudaDeviceSynchronize());   // Wait until GPU kernel function finishes !!
    HANDLE_ERROR(cudaMemcpy(builder_arr, dev_arr, 10 * sizeof(int *), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_arr));
    for (int i = 0; i < 10; i++) {
        HANDLE_ERROR(cudaMemcpy(h_arr[i], builder_arr[i], 10 * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(builder_arr[i]));
    }
    printf("CPU sees h_arr[1][1] = %d\n", h_arr[1][1]);
    for (int i = 0; i > 10; i++) {
        free(h_arr[i]);
    }
    free(h_arr);
}

__global__ void kernel_array2DStructDynamicArr(struct dynamic_array_struct **arr) {
    printf("GPU sees arr[1][1].i[1] = %d\n", arr[1][1].i[1]);
    arr[1][1].i[1] *= 10;
}

void array2D_struct_dynamicArr() {
    struct dynamic_array_struct **h_arr = (struct dynamic_array_struct **) malloc(10 * sizeof(struct dynamic_array_struct *));
    for (int i = 0; i < 10; i++) {
        h_arr[i] = (struct dynamic_array_struct *) malloc(10 * sizeof(struct dynamic_array_struct));
        for (int j = 0; j < 10; j++) {
            h_arr[i][j].i = (int *) malloc(10 * sizeof(int));
            h_arr[i][j].i[1] = i + j;
        }
    }

    struct dynamic_array_struct **Builder1 = (struct dynamic_array_struct **) malloc(10 * sizeof(struct dynamic_array_struct *));
    for (int i = 0; i < 10; i++) {
        Builder1[i] = (struct dynamic_array_struct *) malloc(10 * sizeof(struct dynamic_array_struct));
        for (int j = 0; j < 10; j++) {
            HANDLE_ERROR(cudaMalloc((void **) &Builder1[i][j].i, 10 * sizeof(int)));
            HANDLE_ERROR(cudaMemcpy(Builder1[i][j].i, h_arr[i][j].i, 10 * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    struct dynamic_array_struct **dev_arr;
    HANDLE_ERROR(cudaMalloc((void **) &dev_arr, 10 * sizeof(struct dynamic_array_struct *)));
    struct dynamic_array_struct *builder_arr[10];
    for (int i = 0; i < 10; i++) {
        HANDLE_ERROR(cudaMalloc((void **) &builder_arr[i], 10 * sizeof(struct dynamic_array_struct)));
        HANDLE_ERROR(cudaMemcpy(builder_arr[i], Builder1[i], 10 * sizeof(struct dynamic_array_struct), cudaMemcpyHostToDevice));
    }
    HANDLE_ERROR(cudaMemcpy(dev_arr, builder_arr, 10 * sizeof(struct dynamic_array_struct *), cudaMemcpyHostToDevice));

    kernel_array2DStructDynamicArr<<<1,1>>>(dev_arr);
    HANDLE_ERROR(cudaDeviceSynchronize());   // Wait until GPU kernel function finishes !!

    HANDLE_ERROR(cudaMemcpy(builder_arr, dev_arr, 10 * sizeof(struct dynamic_array_struct *), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_arr));
    for (int i = 0; i < 10; i++) {
        HANDLE_ERROR(cudaMemcpy(Builder1[i], builder_arr[i], 10 * sizeof(struct dynamic_array_struct), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(builder_arr[i]));
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            HANDLE_ERROR(cudaMemcpy(h_arr[i][j].i, Builder1[i][j].i, 10 * sizeof(int), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaFree(Builder1[i][j].i));
        }
    }
    printf("CPU sees h_arr[1][1].i[1] = %d\n", h_arr[1][1].i[1]);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            free(h_arr[i][j].i);
        }
        free(h_arr[i]);
    }
    free(h_arr);
}

int main() {
    array2D_struct_dynamicArr();
    return 0;
}

void HANDLE_ERROR(cudaError_t error) {

}


