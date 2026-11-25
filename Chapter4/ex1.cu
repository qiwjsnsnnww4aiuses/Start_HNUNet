#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error @%d: %s\n", __LINE__, \
               cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

// GPU 0 向 GPU 1 的显存写入
__global__ void remote_store(int* peer, int value) {
    peer[0] = value;
}

// GPU 0 从 GPU 1 的显存读取
__global__ void remote_load(const int* peer) {
    int value = peer[0];
    printf("peer[0]: %d\n", value);
}

static void wait_for_enter(const char* msg) {
    printf("%s\n", msg);
    fflush(stdout);
    getchar();
}

int main(int argc, char** argv) {

    if (argc < 2 ||
        (strcmp(argv[1], "store") != 0 &&
         strcmp(argv[1], "load") != 0)) {
        printf("Usage: %s store|load [--pause]\n", argv[0]);
        return 1;
    }

    bool do_store = (strcmp(argv[1], "store") == 0);
    bool pause = (argc >= 3 && strcmp(argv[2], "--pause") == 0);

    const int test_value = 42;

    // 检查 GPU 0 是否能访问 GPU 1
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, 0, 1));

    printf("GPU 0 can access GPU 1: %s\n",
           can_access ? "yes" : "no");

    if (!can_access) {
        printf("No P2P access. Abort.\n");
        return 1;
    }

    // 在 GPU 1 上申请显存
    int* peer_on_gpu1 = nullptr;

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&peer_on_gpu1, sizeof(int)));

    // Store 模式初始值为 0
    // Load 模式初始值为 42
    const int initial_value = do_store ? 0 : test_value;

    CUDA_CHECK(cudaMemcpy(
        peer_on_gpu1,
        &initial_value,
        sizeof(int),
        cudaMemcpyHostToDevice
    ));

    // 切换到 GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // GPU 0 开启对 GPU 1 的 P2P 访问
    cudaError_t peer_err = cudaDeviceEnablePeerAccess(1, 0);

    if (peer_err != cudaSuccess &&
        peer_err != cudaErrorPeerAccessAlreadyEnabled) {

        printf("Peer access error: %s\n",
               cudaGetErrorString(peer_err));
        return 1;
    }

    if (peer_err == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    }

    printf("\nMode: %s\n",
           do_store ? "REMOTE STORE" : "REMOTE LOAD");

    printf("Kernel runs on GPU 0; buffer resides on GPU 1.\n");

    if (pause) {
        wait_for_enter(
            "[PAUSE 1] Sample NVLink counters, then press Enter."
        );
    }

    // 执行远程写或远程读
    if (do_store) {
        remote_store<<<1, 1>>>(peer_on_gpu1, test_value);
    } else {
        remote_load<<<1, 1>>>(peer_on_gpu1);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (pause) {
        wait_for_enter(
            "[PAUSE 2] Kernel completed. Sample counters again."
        );
    }

    // 验证远程写的结果
    if (do_store) {

        int value_read_back = -1;

        CUDA_CHECK(cudaSetDevice(1));

        CUDA_CHECK(cudaMemcpy(
            &value_read_back,
            peer_on_gpu1,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        printf("GPU 1 buffer after Store: %d (expected %d)\n",
               value_read_back, test_value);

    } else {
        printf("Remote Load completed. See device printf above.\n");
    }

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(peer_on_gpu1));

    return 0;
}