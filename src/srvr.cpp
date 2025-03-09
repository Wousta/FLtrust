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
//#include <torch/torch.h>

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

int main(int argc, char *argv[]) {
  Logger::instance().log("Server starting execution...");

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
  auto result = cli.parse({argc, argv});
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
              << std::endl;
    return 1;
  }

  // mr data and addr
  uint64_t reg_sz = 4096;
  reg_info.addr_locs.push_back(castI(malloc(reg_sz)));
  reg_info.data_sizes.push_back(reg_sz);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());

  
  std::array<int, 3> w = {0, 0, 0};
  for(int i = 0; i < GLOBAL_ITERS; i++) {

    // accept client conn requests and extract info
    RcConn conn;
    int ret = conn.acceptConn(addr_info, reg_info);
    comm_info conn_data = conn.getConnData();

    // 1- copy data to write in your local memory
    std::memcpy(castV(reg_info.addr_locs.front()), w.data(), w.size() * sizeof(int));

    // 2- write msg to remote side
    Logger::instance().log("writing msg ...\n");

    (void)norm::write(conn_data, {w.size() * sizeof(int)}, {LocalInfo()}, NetFlags(),
                      RemoteInfo(), latency, posted_wqes);

    Logger::instance().log("  msg wrote = ");
    for(const auto& i : w) {
      Logger::instance().log(std::to_string(i) + " ");
    }
    Logger::instance().log("\n");

    //clear local memory
    std::memset(castV(reg_info.addr_locs.front()), 0, w.size() * sizeof(int));

    // Training step
    Logger::instance().log("  FLTrust returned: ");
    w[1] = runMNISTTrain();
    for(const auto& i : w) {
      Logger::instance().log(std::to_string(i) + " ");
    }
    Logger::instance().log("\n");

    std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: proper synchronization

    // update weights
    std::array<int, 3> msg;
    std::memcpy(msg.data(), castV(reg_info.addr_locs.front()), w.size() * sizeof(int));
    Logger::instance().log("  Received msg: ");

    for(const auto& i : msg) {
      Logger::instance().log(std::to_string(i) + " ");
    }
    Logger::instance().log("\n");

    // HERE I WOULD AGREGGATE THE WEIGHTS
    for(int i = 0; i < w.size(); i++) {
      w[i] += msg[2];
    }

    Logger::instance().log("  Updated weights: ");
    for(const auto& i : w) {
      Logger::instance().log(std::to_string(i) + " ");

    }
    Logger::instance().log("\n");


    std::this_thread::sleep_for(std::chrono::seconds(1));

  }

  std::cout << "Server done\n";

  // sleep for server to be available
  Logger::instance().log("Sleeping for 1 hour\n");

  std::this_thread::sleep_for(std::chrono::hours(1));
  return 0;
}