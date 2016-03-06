
#include <stdio.h>




__global__ void add(int *a, int *b, int *c){
     *c = *a + *b;
}


extern "C" {
    int test_addition(void) {

        printf("CUDA status: %d\n", cudaDeviceSynchronize());

        int a, b, c; // host copies of a, b, c
            int *d_a, *d_b, *d_c; // device copies of a, b, c
            int size = sizeof(int);
            // Allocate space for device copies of a, b, c
            cudaMalloc((void **)&d_a, size);
            cudaMalloc((void **)&d_b, size);
            cudaMalloc((void **)&d_c, size);
            // Setup input values
            a = 1;
            b = 7;

            // Copy inputs to device
            cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
            // Launch add() kernel on GPU
            add<<<1,1>>>(d_a, d_b, d_c);
            // Copy result back to host
            cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
            // Cleanup
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

             printf("Result is: %d\n", c);


        return c;
     }
 }

 extern "C" {
     void count_devices(void) {
       int nDevices;

       cudaGetDeviceCount(&nDevices);
       for (int i = 0; i < nDevices; i++) {
         cudaDeviceProp prop;
         cudaGetDeviceProperties(&prop, i);
         printf("Device Number: %d\n", i);
         printf("  Device name: %s\n", prop.name);
         printf("  Memory Clock Rate (KHz): %d\n",
                prop.memoryClockRate);
         printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
         printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
       }
     }
 }