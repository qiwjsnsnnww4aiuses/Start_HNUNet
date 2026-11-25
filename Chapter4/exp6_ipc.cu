#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#define CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        printf("CUDA 错误 @%d: %s\n", __LINE__, cudaGetErrorString(e_)); \
        return 1; \
    } \
} while(0)

const int PORT = 29777;
__global__ void write_peer(int* peer, int val){
    peer[0] = val;
}

static int read_full(int fd, void* buf, int n){
    int  got = 0;
    while(got<n){
        int r = recv(fd, (int*)buf+got, n-got, 0);
        got += r;
    }
    return 0;
}

int holder(){
    int* buf1 = nullptr;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&buf1, sizeof(int)));
    CHECK(cudaMemset(buf1, 0, sizeof(int)));
    printf("[holder pid=%d] 卡 1 显存就绪，本进程指针 buf = %p\n", getpid(), (void*)buf1);
    cudaIpcMemHandle_t handle;
    CHECK(cudaIpcGetMemHandle(&handle, buf1));
    printf("[holder] 句柄共 %zu 字节，前16字节：", sizeof(handle));
    for(int i = 0; i < 16; ++i)
        printf("%02x", ((unsigned char*)&handle)[i]);
    printf("...\n");

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(PORT);
    if(bind(srv, (sockaddr*)&addr, sizeof(addr))||listen(srv, 1)){
        perror("bind/listen"); return 1;
    }

    printf("[holder] 等待 writer 连接(端口 %d) ..\n", PORT);
    int conn = accept(srv, nullptr, nullptr);
    if(conn < 0) { perror("accept"); return 1;}
    send(conn, &handle, sizeof(handle), 0);
    printf("[holder] 句柄已发出， 等 writer 的回执...\n");
    
    char ack;
    if(read_full(conn, &ack, 1)) {printf("回执读入失败\n"); return 1;}
    int h = -1;
    CHECK(cudaMemcpy(&h, buf1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[holder]回读自己卡 1 的显存: %d(应为42--writer_kernel写入)", h);
    return 0;
}

int writer(){
    //使用方：领卡，开门，直写
    int conn = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    if (connect(conn, (sockaddr*)&addr, sizeof(addr))){
        perror("connect (请先在另一个终端启动 holder)"); return 1;
    }

    cudaIpcMemHandle_t handle;
    if (read_full(conn, &handle, sizeof(handle))) {printf("句柄接受失败\n"); return 1;}
    printf("[writer pid=%d] 收到 %zu 字节句柄\n", getpid(), sizeof(handle));
    CHECK(cudaSetDevice(0));
    int* p = nullptr;
    CHECK(cudaIpcOpenMemHandle((void**)&p, handle, cudaIpcMemLazyEnablePeerAccess));
    printf("[writer] 开门成功， 本进程指针 p = %p\n", (void*)p);
    write_peer<<<1, 1>>>(p, 42);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    printf("[writer] 已写入，回执 holder\n");
    char ack = 'K';
    send(conn, &ack, 1, 0);
    return 0;
}


int main(int argc, char** argv){
    if(argc==2 && !strcmp(argv[1], "holder")) return holder();
    if(argc==2 && !strcmp(argv[1], "writer")) return writer();
    return 1;
}
