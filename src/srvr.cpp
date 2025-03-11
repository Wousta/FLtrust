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
  std::vector<RegInfo> reg_info(n_clients);
  std::vector<RcConn> conns(n_clients);
  std::vector<comm_info> conn_data;
  std::vector<LocalInfo> loc_info(n_clients);
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

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());

  // mr data and addr
  uint64_t reg_sz_data = REG_SZ_DATA;
  uint64_t step_sz = sizeof(uint64_t);

  for(int i = 0; i < n_clients; i++) {
    reg_info[i].addr_locs.push_back(castI(malloc(step_sz)));
    reg_info[i].addr_locs.push_back(castI(malloc(reg_sz_data)));
    reg_info[i].data_sizes.push_back(step_sz);
    reg_info[i].data_sizes.push_back(reg_sz_data);
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    conns[i].acceptConn(addr_info, reg_info[i]);
    conn_data.push_back(conns[i].getConnData());

    loc_info[i].offs.push_back(0);
    loc_info[i].offs.push_back(0);
    loc_info[i].indices.push_back(1);
  }


  // Create a dummy set of weights, needed for first call to runMNISTTrain():
  std::vector<torch::Tensor> w_dummy;
  w_dummy.push_back(torch::randn({10}, torch::kFloat32)); 
  //std::vector<torch::Tensor> w = runMNISTTrain();

  std::vector<torch::Tensor> w = runMNISTTrainDummy(w_dummy);
  for(int i = 0; i < GLOBAL_ITERS; i++) {

    // Flatten and concatenate all parameters into one contiguous tensor.
    auto w_flat = torch::cat(w).contiguous();
    size_t total_bytes = w_flat.numel() * sizeof(float);
    float* raw_ptr = w_flat.data_ptr<float>();

    // Send the number of elements in the tensor to the client.
    int64_t numel = w_flat.numel();

    std::cout << "number of elements in w_flat server: " << w_flat.numel() << std::endl;
    std::cout << "total bytes: " << total_bytes << std::endl;

    // // 2- write msg to remote side
    Logger::instance().log("writing msg ...\n");

    unsigned int total_bytes_int = static_cast<unsigned int>(total_bytes);
    for(int i = 0; i < n_clients; i++) {
      std::memcpy(castV(reg_info[i].addr_locs[1]), raw_ptr, total_bytes);
      (void)norm::send(conn_data[i], {total_bytes_int}, {loc_info[i]}, NetFlags(), latency, posted_wqes);
    }

    std::cout << "server sent w to clients\n";

    std::vector<torch::Tensor> g = runMNISTTrainDummy(w);

    // TODO: proper synch of client sending back weights
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // // Read the weights sent by the client
    // size_t total_bytes_g = reg_info.data_sizes[1];
    // size_t numel_server = total_bytes_g / sizeof(float);
    // float* client_data = static_cast<float*>(castV(reg_info.addr_locs[1]));

    // // Create a tensor from the raw data (and clone it to own its memory)
    // auto updated_tensor = torch::from_blob(client_data, {static_cast<long>(numel_server)}, torch::kFloat32).clone();

    // std::vector<torch::Tensor> g_client = {updated_tensor};

    // // For verification, print the first few updated weight values.
    // std::cout << "Updated weights from client:" << std::endl;
    // std::cout << g_client[0].slice(0, 0, std::min<size_t>(g_client[0].numel(), 10)) << std::endl;
    
  }

  std::cout << "Server done\n";

  // sleep for server to be available
  Logger::instance().log("Sleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}