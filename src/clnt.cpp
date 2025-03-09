#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include "include/globalConstants.hpp"

#include <logger.hpp>
#include <lyra/lyra.hpp>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

int main(int argc, char *argv[]) {
  int id = std::stoi(argv[argc -1]);
  Logger::instance().log("Client starting execution with id: " + std::to_string(id) + "\n");

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
  uint64_t reg_sz_data = 87392;
  reg_info.addr_locs.push_back(castI(malloc(reg_sz_data)));
  reg_info.data_sizes.push_back(reg_sz_data);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());


  std::vector<torch::Tensor> w = runMNISTTrainDummy();
  for(int i = 0; i < GLOBAL_ITERS; i++) {
    // connect to server
    RcConn conn;

    Logger::instance().log("Connecting to server W\n");
    int ret = conn.connect(addr_info, reg_info);
    comm_info conn_data = conn.getConnData();

    std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: proper synchronization
    
    // 2- Get number of elements in the tensor from the server
    // int64_t numel;
    // std::memcpy(&numel, castV(reg_info.addr_locs.front()), sizeof(numel));

    // std::cout << "number of elements in w_flat client: " << numel << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: proper synchronization

    // Read the updated weights from the server
    size_t total_bytes = reg_info.data_sizes.front();
    size_t numel_server = total_bytes / sizeof(float);
    float* server_data = static_cast<float*>(castV(reg_info.addr_locs.front()));

    // Option 2: Create a tensor from the raw data (and clone it to own its memory)
    auto updated_tensor = torch::from_blob(server_data, {static_cast<long>(numel_server)}, torch::kFloat32).clone();

    std::cout << "Number of elements in updated tensor: " << updated_tensor.numel() << std::endl;
    // For verification, print the first few updated weight values.
    std::cout << "Updated weights from server:" << std::endl;
    std::cout << updated_tensor.slice(0, 0, std::min(numel_server, size_t(10))) << std::endl;

    // Send updated model to server
    // std::memcpy(castV(reg_info.addr_locs.front()), w.data(), w.size() * sizeof(int));
    // Logger::instance().log("writing msg ...\n");

    // (void)norm::write(conn_data, {w.size() * sizeof(int)}, {LocalInfo()}, NetFlags(),
    //                   RemoteInfo(), latency, posted_wqes);

    // Logger::instance().log("  msg wrote =");

    // for(const auto& i : w) {
    //   Logger::instance().log(std::to_string(i) + " ");
    // }
    // Logger::instance().log("\n");

  }

  std::cout << "Client done\n";

  return 0;
}
