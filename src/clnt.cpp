#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include "include/globalConstants.hpp"

#include <logger.hpp>
#include <lyra/lyra.hpp>
//#include <torch/torch.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

std::array<int, 3> modelUpdate(std::array<int, 3> w) {
  // Training step
  std::cout << "FLTrust returned: ";
  w[1] = runMNISTTrain();
  for(const auto& i : w) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  return w;
}

int main(int argc, char *argv[]) {
  Logger::instance().log("Client starting execution...");

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

  // mr data
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
    // connect to server
    RcConn conn;

    Logger::instance().log("Connecting to server W\n");
    int ret = conn.connect(addr_info, reg_info);
    comm_info conn_data = conn.getConnData();

    std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: proper synchronization
    
    // 2- verify local mem
    std::array<int, 3> msg;
    std::memcpy(msg.data(), castV(reg_info.addr_locs.front()), w.size() * sizeof(int));

    // Training step
    Logger::instance().log("FLTrust returned:");

    w[0] = runMNISTTrain();
    w[1] = msg[1];
    w[2] = w[0] + w[1];
    for(const auto& i : w) {
      Logger::instance().log(std::to_string(i) + " ");
    }
    Logger::instance().log("\n");

    // Send updated model to server
    std::memcpy(castV(reg_info.addr_locs.front()), w.data(), w.size() * sizeof(int));
    Logger::instance().log("writing msg ...\n");

    (void)norm::write(conn_data, {w.size() * sizeof(int)}, {LocalInfo()}, NetFlags(),
                      RemoteInfo(), latency, posted_wqes);

    Logger::instance().log("  msg wrote =");

    for(const auto& i : w) {
      Logger::instance().log(std::to_string(i) + " ");
    }
    Logger::instance().log("\n");

  }

  std::cout << "Client done\n";

  return 0;
}
