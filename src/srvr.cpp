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

  // stores the parameters W of the server, to be read by clients
  float* srvr_w =  reinterpret_cast<float*> (malloc(REG_SZ_DATA));

  // Flag that server modifies so clients can start reading
  std::atomic<uint64_t>* cas_atomic = new std::atomic<uint64_t>(0);

  for (int i = 0; i < n_clients; i++) {
    std::atomic<uint64_t>* cas_client = new std::atomic<uint64_t>(0);
    reg_info[i].addr_locs.push_back(castI(cas_atomic));
    reg_info[i].addr_locs.push_back(castI(srvr_w));
    //reg_info[i].addr_locs.push_back(castI(malloc(REG_SZ_DATA)));
    //reg_info[i].addr_locs.push_back(castI(cas_client));
    reg_info[i].data_sizes.push_back(CAS_SIZE);
    reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    // reg_info[i].data_sizes.push_back(REG_SZ_DATA);
    // reg_info[i].data_sizes.push_back(CAS_SIZE);
    reg_info[i].permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    conns[i].acceptConn(addr_info, reg_info[i]);
    conn_data.push_back(conns[i].getConnData());

    loc_info[i].offs.push_back(0);
    loc_info[i].offs.push_back(0);
    loc_info[i].indices.push_back(0);
    loc_info[i].indices.push_back(1);
  }


  // Create a dummy set of weights, needed for first call to runMNISTTrain():
  std::vector<torch::Tensor> w_dummy;
  w_dummy.push_back(torch::randn({10}, torch::kFloat32)); 
  //std::vector<torch::Tensor> w = runMNISTTrain();

  std::vector<torch::Tensor> w = runMNISTTrainDummy(w_dummy);
  for (int i = 0; i < GLOBAL_ITERS; i++) {

    // Store w in shared memory
    auto all_tensors = torch::cat(w).contiguous();
    size_t total_bytes = all_tensors.numel() * sizeof(float);
    std::memcpy(srvr_w, all_tensors.data_ptr<float>(), total_bytes);

    std::cout << "Server wrote bytes = " << total_bytes << "\n";
    // Print a slice of the weights
    {
      std::ostringstream oss;
      oss << "Updated weights from server:" << "\n";
      oss << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << " ";
      oss << "...\n";
      Logger::instance().log(oss.str());
    }

    uint64_t expected = 0;
    if (cas_atomic->compare_exchange_strong(expected, 1)) {
      std::cout << "CAS succeeded: value set to 1\n";
    } else {
        std::cout << "CAS failed: current value = " << cas_atomic->load() << "\n";
    }

    // std::vector<torch::Tensor> g = runMNISTTrainDummy(w);

    // int clnts_finished = 0;
    // while (clnts_finished != n_clients) {
    //   clnts_finished = 0;
    //   for (int i = 0; i < n_clients; i++) {

    //     std::atomic<uint64_t>* cas_client_ptr =  
    //           (std::atomic<uint64_t>*) castV(reg_info[i].addr_locs[2]);
    //     uint64_t client_cas_val = cas_client_ptr->load();

    //     if (client_cas_val == 1) {
    //       clnts_finished++;
    //     }

    //   }
    // }

    // expected = 1;
    // if (cas_atomic->compare_exchange_strong(expected, 0)) {
    //   std::cout << "CAS succeeded at END: value set to 0\n";
    // } else {
    //     std::cout << "CAS failed at END: current value = " << cas_atomic->load() << "\n";
    // }
    
  }

  std::cout << "Server done\n";

  // sleep for server to be available
  Logger::instance().log("Sleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}