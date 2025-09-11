# FLtrust (C++/LibTorch + RDMA)

Prototype implementation of FLTrust, a robust federated learning aggregation method that down-weights untrustworthy client updates using a small trusted server dataset. Based on “FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping” (arXiv:2012.13995).

## Description

This repository contains a server–client prototype of FLTrust built in C++ on top of LibTorch (PyTorch C++ API) and an RDMA transport layer. The server broadcasts model weights to clients; clients perform local training and send updates back. The server would compute trust scores by comparing each client’s update to a trusted gradient derived from a clean server dataset, and aggregate accordingly.

Current status:

-   Networking: custom RDMA layer via external libraries.
-   Model/training: MNIST CNN defined in `src/include/mnistTrain.hpp` with a placeholder training path (`runMNISTTrainDummy`) used for scaffolding. Swap to `runMNISTTrain` for real training with the MNIST dataset.

Paper reference (summary): FLTrust assigns each client a trust score based on the cosine similarity between the client update and the trusted server gradient, truncates negative contributions, and aggregates normalized updates to resist Byzantine clients.

## Getting Started

### Dependencies

-   OS: Linux x86_64 recommended (Ubuntu 22.04+). Windows users: use WSL2 Ubuntu for build/run. Native Windows is not supported by the RDMA stack used here.
-   Toolchain: CMake ≥ 3.22, Conan (v2), g++-12 (or Clang with C++17), Git.
-   LibTorch: Install the C++ distribution (CPU or CUDA). You must set `Torch_DIR` to the LibTorch CMake package folder (e.g., `<libtorch>/share/cmake/Torch`).
-   RDMA stack: rdma-core/ibverbs and a RoCE/InfiniBand RNIC for real runs. Local build without RNIC is possible but network ops will fail at runtime.
-   Redis (optional): used only by `run_srvr.sh` as a convenience for simple coordination/log resets.
-   External sibling projects (must be cloned and built next to this repo):
    -   `../RcConn` → supplies `librc_conn.so`
    -   `../rdma-api` → supplies `libnorm.so`
    -   A small shared utilities repo providing `shared/util.hpp` (expected at `../shared/util.hpp` from this repo).

The CMake file links to:

-   `../RcConn/build/librc_conn.so`
-   `../rdma-api/build/libnorm.so`

Make sure those paths exist after building the sibling projects.

### Installing / Building

You can use the provided script on Linux, or run the equivalent commands manually.

-   Linux (bash):

    1. Install prerequisites (CMake, Conan, compiler, rdma-core, Redis optional).
    2. Ensure `Torch_DIR` is set correctly (env or CMake cache).
    3. Build sibling repos `RcConn` and `rdma-api` in `../RcConn/build` and `../rdma-api/build`.
    4. From this repo:
        - `./build.sh` (edits may be needed for your compiler and `Torch_DIR`).

-   Windows:
    -   Use WSL2 Ubuntu. Build exactly as in Linux instructions. Native Windows builds are not supported due to RDMA and POSIX-specific dependencies.

Notes:

-   `CMakeLists.txt` currently hard-codes `/usr/bin/g++-12`. If your compiler is elsewhere, edit or remove that line.
-   The preload `cache.cmake` expects a Linux path for `Torch_DIR`; override with your local LibTorch path.

### Executing program

Two binaries are produced in `build/`: `srvr` and `clnt`.

-   Start the server (example):

    -   `build/srvr --srvr_ip 192.168.1.10 --port 2000 --n_clients 1`

-   Start a client (example):
    -   `build/clnt --srvr_ip 192.168.1.10 --port 2000 --id 1`

Arguments:

-   `--srvr_ip`: IPv4 of the host running the server.
-   `--port`: port for the control/handshake layer used by the RDMA connection helper.
-   `--n_clients` (server only): how many clients to expect.
-   `--id` (client only): client identifier.

Convenience scripts (Linux):

-   `run_srvr.sh` – starts Redis (optional) and launches the server and clients. Adjust `srvr_ip`, `port`, and `n_clients` inside the script.
-   `run_client.sh` – simple wrapper to run a single client.

Logs:

-   Each process writes to `logs/execution_<pid>.log` via the `Logger` singleton.

Data:

-   Real MNIST training (`runMNISTTrain`) uses LibTorch’s dataset API. The dataset will be downloaded to `./data` by default if network is available.
-   The current prototype uses `runMNISTTrainDummy` which simply perturbs weights to exercise the end-to-end path.

Configuration knobs (see `src/include/globalConstants.hpp`):

-   `GLOBAL_ITERS`: federated rounds to run.
-   `REG_SZ_DATA`: size (bytes) of the RDMA-registered weight buffer. Must be ≥ total flattened model size (sum of parameter numel × sizeof(float)). Increase if you switch to a larger model.

## Help

Common issues and fixes:

-   Torch not found / bad `Torch_DIR`:

    -   Set `-DTorch_DIR=<path-to-libtorch>/share/cmake/Torch` when configuring CMake, or edit `cache.cmake`.

-   Wrong compiler path:

    -   Edit or remove the `set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")` line in `CMakeLists.txt`.

-   Missing sibling libs (`librc_conn.so`, `libnorm.so`):

    -   Build `../RcConn` and `../rdma-api` so files exist at the paths CMake links to, or update the link paths.

    -   Ensure rdma-core is installed, the RDMA service is up, and you have a supported RNIC and proper IP/roce/IB configuration.

-   Windows build errors:

    -   Use WSL2 Ubuntu. This repo uses POSIX APIs (e.g., `unistd.h`) and RDMA libraries unavailable on native Windows.

-   Buffer too small:
    -   If you switch to real training and see partial weights or memory issues, increase `REG_SZ_DATA` to cover the flattened parameter bytes.

## Version History

-   0.1
    -   First public prototype: RDMA server/client scaffold, MNIST model, dummy local train path.

## License

No license file is included at this time. All rights reserved by the author. Contact the author to discuss licensing.

## Acknowledgments

-   FLTrust paper: https://arxiv.org/pdf/2012.13995
-   LibTorch (PyTorch C++ API)
-   cli11/lyra library for CLI parsing (via Conan)
-   RDMA stack: rdma-core / ibverbs
-   Redis (optional helper in scripts)

##

Author: Luis Bustamante
