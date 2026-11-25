#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda_runtime.h>

const int PORT = 29777;
const int VALUE_TO_WRITE = 12345;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error @%d: %s\n", __LINE__, \
               cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)


// 发送完整的 n 字节
static int send_all(int fd, const void* buffer, size_t n) {

    const char* p = (const char*)buffer;
    size_t sent = 0;

    while (sent < n) {

        ssize_t r = send(
            fd,
            p + sent,
            n - sent,
            MSG_NOSIGNAL
        );

        if (r < 0) {
            if (errno == EINTR)
                continue;

            return -1;
        }

        if (r == 0)
            return -1;

        sent += (size_t)r;
    }

    return 0;
}


// 接收完整的 n 字节
static int recv_full(int fd, void* buffer, size_t n) {

    char* p = (char*)buffer;
    size_t received = 0;

    while (received < n) {

        ssize_t r = recv(
            fd,
            p + received,
            n - received,
            0
        );

        if (r == 0)
            return -1;

        if (r < 0) {
            if (errno == EINTR)
                continue;

            return -1;
        }

        received += (size_t)r;
    }

    return 0;
}


// GPU Kernel：向远程显存写入
__global__ void write_peer(int* peer, int value) {
    peer[0] = value;
}


// ==========================================
// HOLDER
// ==========================================

int holder() {

    int* buf1 = nullptr;

    int srv = -1;
    int conn = -1;

    // 在 GPU 1 上申请显存
    CUDA_CHECK(cudaSetDevice(1));

    CUDA_CHECK(cudaMalloc(&buf1, sizeof(int)));

    CUDA_CHECK(cudaMemset(buf1, 0, sizeof(int)));

    printf("[holder] GPU 1 memory ready: %p\n",
           (void*)buf1);


    // 导出 CUDA IPC Handle
    cudaIpcMemHandle_t handle;

    CUDA_CHECK(cudaIpcGetMemHandle(&handle, buf1));


    // 创建 TCP Server
    srv = socket(AF_INET, SOCK_STREAM, 0);

    if (srv < 0) {
        perror("socket");
        cudaFree(buf1);
        return 1;
    }


    // 设置端口复用
    int one = 1;

    if (setsockopt(
            srv,
            SOL_SOCKET,
            SO_REUSEADDR,
            &one,
            sizeof(one)) < 0) {

        perror("setsockopt");
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    // 设置服务器地址
    sockaddr_in addr{};

    addr.sin_family = AF_INET;

    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    addr.sin_port = htons(PORT);


    // bind
    if (bind(
            srv,
            (sockaddr*)&addr,
            sizeof(addr)) < 0) {

        perror("bind");
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    // listen
    if (listen(srv, 1) < 0) {

        perror("listen");
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    printf("[holder] Waiting for writer...\n");


    // accept
    conn = accept(srv, nullptr, nullptr);

    if (conn < 0) {

        perror("accept");
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    // holder 决定 writer 要写入什么
    const int value = VALUE_TO_WRITE;


    // 第一次发送：CUDA IPC Handle
    if (send_all(
            conn,
            &handle,
            sizeof(handle)) != 0) {

        perror("send handle");
        close(conn);
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    // 第二次发送：写入值
    if (send_all(
            conn,
            &value,
            sizeof(value)) != 0) {

        perror("send value");
        close(conn);
        close(srv);
        cudaFree(buf1);
        return 1;
    }


    printf("[holder] Sent handle: %zu bytes\n",
           sizeof(handle));

    printf("[holder] Sent value: %d\n", value);


    // 等待 writer 的应用层 ACK
    char ack = '\0';

    if (recv_full(conn, &ack, 1) != 0) {

        printf("[holder] ACK receive failed\n");

        close(conn);
        close(srv);
        cudaFree(buf1);

        return 1;
    }

    if (ack != 'K') {

        printf("[holder] Invalid ACK\n");

        close(conn);
        close(srv);
        cudaFree(buf1);

        return 1;
    }


    // 从 GPU 1 回读显存
    int result = -1;

    CUDA_CHECK(cudaMemcpy(
        &result,
        buf1,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));


    printf("[holder] GPU 1 memory = %d\n", result);

    printf("[holder] Expected = %d\n", value);


    close(conn);
    close(srv);

    CUDA_CHECK(cudaFree(buf1));

    return (result == value) ? 0 : 1;
}



// ==========================================
// WRITER
// ==========================================

int writer() {

    // 创建客户端 Socket
    int conn = socket(AF_INET, SOCK_STREAM, 0);

    if (conn < 0) {
        perror("socket");
        return 1;
    }


    // 设置服务器地址
    sockaddr_in addr{};

    addr.sin_family = AF_INET;

    addr.sin_port = htons(PORT);

    // holder 和 writer 在同一台机器
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);


    // connect
    if (connect(
            conn,
            (sockaddr*)&addr,
            sizeof(addr)) < 0) {

        perror("connect");
        close(conn);
        return 1;
    }


    // 接收 Handle
    cudaIpcMemHandle_t handle;

    if (recv_full(
            conn,
            &handle,
            sizeof(handle)) != 0) {

        printf("[writer] Handle receive failed\n");

        close(conn);
        return 1;
    }


    // 接收 holder 指定的写入值
    int value = 0;

    if (recv_full(
            conn,
            &value,
            sizeof(value)) != 0) {

        printf("[writer] Value receive failed\n");

        close(conn);
        return 1;
    }


    printf("[writer] Received handle: %zu bytes\n",
           sizeof(handle));

    printf("[writer] Received value: %d\n", value);


    // writer 在 GPU 0 上执行 Kernel
    CUDA_CHECK(cudaSetDevice(0));


    // 打开 holder 导出的显存
    int* p = nullptr;

    CUDA_CHECK(cudaIpcOpenMemHandle(
        (void**)&p,
        handle,
        cudaIpcMemLazyEnablePeerAccess
    ));


    printf("[writer] IPC Open success: %p\n", (void*)p);


    // 执行 Kernel
    write_peer<<<1, 1>>>(p, value);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());


    printf("[writer] Kernel completed\n");


    // 关闭显存映射
    CUDA_CHECK(cudaIpcCloseMemHandle(p));


    // 发送应用层 ACK
    char ack = 'K';

    if (send_all(conn, &ack, 1) != 0) {

        perror("send ACK");

        close(conn);
        return 1;
    }


    printf("[writer] ACK sent\n");

    close(conn);

    return 0;
}



// ==========================================
// MAIN
// ==========================================

int main(int argc, char** argv) {

    if (argc != 2) {
        printf("Usage: %s holder|writer\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "holder") == 0)
        return holder();

    if (strcmp(argv[1], "writer") == 0)
        return writer();

    printf("Unknown mode\n");

    return 1;
}