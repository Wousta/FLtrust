#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include "include/training.hpp"
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

    std::cout << "Accepting server W\n";
    int ret = conn.acceptConn(addr_info, reg_info);

    std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: proper synchronization
    
    // 2- verify local mem
    std::array<int, 3> msg;
    std::memcpy(msg.data(), castV(reg_info.addr_locs.front()), w.size() * sizeof(int));

    // Training step
    std::cout << "FLTrust returned: ";
    w[1] = runMNISTTrain();
    w[0] = msg[0];
    for(const auto& i : w) {
      std::cout << i << " ";
    }
    std::cout << "\n";


    // Clear local memory
    std::memset(castV(reg_info.addr_locs.front()), 0, msg.size());

    // int ret = conn.connect(addr_info, reg_info);

    // // extract conn Info
    // comm_info conn_data = conn.getConnData();

    // // Write Test
    // // 1- copy data to write in your local memory
    // std::string msg = "Hello! " + std::to_string(w++);
    // std::memcpy(castV(reg_info.addr_locs.front()), msg.data(), msg.length());

    // // 2- write msg to remote side
    // std::cout << "writing msg ...\n";
    // (void)norm::write(conn_data, {msg.length()}, {LocalInfo()}, NetFlags(),
    //                   RemoteInfo(), latency, posted_wqes);
    // std::cout << "msg wrote = " << msg << "\n";

    // // clear local memory
    // std::memset(castV(reg_info.addr_locs.front()), 0, msg.length());

    // // Read Test
    // // 1- read remote info to ur local mem
    // std::cout << "reading msg ...\n";
    // (void)norm::read(conn_data, {msg.length()}, {LocalInfo()}, NetFlags(),
    //                 RemoteInfo(), latency, posted_wqes);
    // // 2- verify local mem
    // msg.clear();
    // msg.resize(MSG_SZ);
    // std::memcpy(msg.data(), castV(reg_info.addr_locs.front()), msg.length());
    // std::cout << "msg read = " << msg << "\n";

    //std::this_thread::sleep_for(std::chrono::seconds(1));

  }

  return 0;
}
