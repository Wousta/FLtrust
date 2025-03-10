#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include "include/globalConstants.hpp"
#include <logger.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <lyra/lyra.hpp>
#include <string>
#include <thread>
#include <torch/torch.h>

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;


int main(int argc, char *argv[]) {
  int n_clients = std::stoi(argv[argc -1]);
  Logger::instance().log("Server starting execution with id: " + std::to_string(0) + "\n");

  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  RegInfo reg_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();

  latency->reserve(10);
  auto cli = lyra::cli() |
             lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
             lyra::opt(port, "port")["-p"]["--port"]("port");
  auto result = cli.parse({argc - 1, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
              << std::endl;
    return 1;
  }

  // mr data and addr
  uint64_t reg_sz_data = REG_SZ_DATA;
  reg_info.addr_locs.push_back(castI(malloc(reg_sz_data)));
  reg_info.data_sizes.push_back(reg_sz_data);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  
  // Create a dummy set of weights, needed for first call to runMNISTTrain():
  std::vector<torch::Tensor> w_dummy;
  w_dummy.push_back(torch::randn({10}, torch::kFloat32)); 
  //std::vector<torch::Tensor> w = runMNISTTrain();

  std::vector<torch::Tensor> w = runMNISTTrainDummy(w_dummy);
  for(int i = 0; i < GLOBAL_ITERS; i++) {
    // accept client conn requests and extract info
    RcConn conn;
    int ret = conn.acceptConn(addr_info, reg_info);
    comm_info conn_data = conn.getConnData();

    // Flatten and concatenate all parameters into one contiguous tensor.
    auto w_flat = torch::cat(w).contiguous();
    size_t total_bytes = w_flat.numel() * sizeof(float);
    float* raw_ptr = w_flat.data_ptr<float>();

    // Send the number of elements in the tensor to the client.
    int64_t numel = w_flat.numel();

    std::cout << "number of elements in w_flat server: " << w_flat.numel() << std::endl;
    std::cout << "total bytes: " << total_bytes << std::endl;

    // // 1- copy data to write in your local memory
    std::memcpy(castV(reg_info.addr_locs.front()), raw_ptr, total_bytes);

    // // 2- write msg to remote side
    Logger::instance().log("writing msg ...\n");

    unsigned int total_bytes_int = static_cast<unsigned int>(total_bytes);
    (void)norm::write(conn_data, {total_bytes_int}, {LocalInfo()}, NetFlags(),
                      RemoteInfo(), latency, posted_wqes);

    std::vector<torch::Tensor> g = runMNISTTrainDummy(w);

    // Read the weights sent by the client
    size_t total_bytes_g = reg_info.data_sizes.front();
    size_t numel_server = total_bytes_g / sizeof(float);
    float* client_data = static_cast<float*>(castV(reg_info.addr_locs.front()));

    // Create a tensor from the raw data (and clone it to own its memory)
    auto updated_tensor = torch::from_blob(client_data, {static_cast<long>(numel_server)}, torch::kFloat32).clone();

    std::vector<torch::Tensor> g_client = {updated_tensor};

    // For verification, print the first few updated weight values.
    std::cout << "Updated weights from client:" << std::endl;
    std::cout << g_client[0].slice(0, 0, std::min<size_t>(g_client[0].numel(), 10)) << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

  }

  std::cout << "Server done\n";

  // sleep for server to be available
  Logger::instance().log("Sleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}