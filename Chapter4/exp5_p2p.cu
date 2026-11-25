// exp5_p2p.cu -- 不开NCCL:一条store指令穿过NVLINK
#include <cstdio>
#include <cuda_runtime.h>
#define CHECK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        printf("CUDA 错误 @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
        return 1;}} while(0)
__global__ void write_peer(int* peer, int v){
    peer[0] = v;
}    

int main(){
    int can = 0;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    printf("cudaDeviceAccessPeer(0+1) = %d (1 = 有P2P通路)\n", can);
    if(!can) {printf("两卡间无P2P通路，本实验退出（见附录A,exp6同）\n");}


    int* buf1 = nullptr;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&buf1, sizeof(int)));
    CHECK(cudaMemset(buf1, 0, sizeof(int)));

    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    write_peer<<<1, 1>>>(buf1, 42);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());


    int h = -1;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMemcpy(&h, buf1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("卡 1 显存里的值 - %d (应为42，且全程没有经过卡0的显存)\n");
    return 0;
}



#include<cstdio>
#include<cuda_runtime.h>
#define CHECK(x) do { cudaError_t e_ = (x); \
    if(e_!=cudaSuccess)
}