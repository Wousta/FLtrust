#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <lyra/lyra.hpp>
#include <string>
#include <thread>
#include <torch/torch.h>

int main(int argc, char *argv[]) {

  
  std::cout << "Trying fltrust\n";
  int retMNIST = runMNISTTrain();
  std::cout << "FLTrust returned: " << retMNIST << "\n";

  std::string srvr_ip;
  std::string port;
  AddrInfo addr_info;
  RegInfo reg_info;
  auto cli = lyra::cli() |
             lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
             lyra::opt(port, "port")["-p"]["--port"]("port");
  auto result = cli.parse({argc, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
              << std::endl;
    return 1;
  }
  // mr data
  uint64_t reg_sz = 4096;
  reg_info.addr_locs.push_back(castI(malloc(reg_sz)));
  reg_info.data_sizes.push_back(reg_sz);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());
  
  while (true) {
    // accept client conn requests
    RcConn conn;
    std::cout << "Accepting clients requests\n";

    int ret = conn.acceptConn(addr_info, reg_info);
    if (ret == 0) {
      std::cout << "ret 0\n";
    }
  }

  // sleep for server to be available
  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}