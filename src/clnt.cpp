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

int main(int argc, char* argv[]) {
  Logger::instance().log("Client starting execution\n");

  int id;
  std::string srvr_ip;
  std::string port;
  unsigned int posted_wqes;
  AddrInfo addr_info;
  RegInfo reg_info;
  std::shared_ptr<ltncyVec> latency = std::make_shared<ltncyVec>();
  latency->reserve(10);
  auto cli = lyra::cli() |
    lyra::opt(srvr_ip, "srvr_ip")["-i"]["--srvr_ip"]("srvr_ip") |
    lyra::opt(port, "port")["-p"]["--port"]("port") |
    lyra::opt(id, "id")["-p"]["--id"]("id");
  auto result = cli.parse({ argc, argv });
  if (!result) {
    std::cerr << "Error in command line: " << result.errorMessage()
      << std::endl;
    return 1;
  }

  // mr data and addr
  //std::atomic<uint64_t>* cas_atomic = new std::atomic<uint64_t>(0);
  int flag = 0;
  float* srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  reg_info.addr_locs.push_back(castI(&flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.data_sizes.push_back(sizeof(flag));
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // addr
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());

  // connect to server
  RcConn conn;

  Logger::instance().log("Connecting to serverrrr W\n");
  int ret = conn.connect(addr_info, reg_info);
  comm_info conn_data = conn.getConnData();

  std::vector<torch::Tensor> w;
  int i = 0;
  for (i = 0; i < GLOBAL_ITERS; i++) {

    std::cout << "Client gonna read flag = " << flag << "\n";

    LocalInfo flag_info;
    flag_info.offs.push_back(0);
    flag_info.indices.push_back(0);
    int ret = 0;
    do {
      ret = norm::read(conn_data, { 0 }, { flag_info }, NetFlags(),
        RemoteInfo(), latency, posted_wqes);
    } while (flag != 1);

    std::cout << "Client read ret = " << ret << "\n";

    ret = norm::read(conn_data, { 0 }, { flag_info }, NetFlags(),
      RemoteInfo(), latency, posted_wqes);

    std::cout << "Client read flag TWO = " << ret << "\n";

    // Read the weights from the server
    LocalInfo data_info;
    data_info.offs.push_back(0);
    data_info.indices.push_back(1);
    (void)norm::read(conn_data, { REG_SZ_DATA }, { data_info }, NetFlags(),
      RemoteInfo(), latency, posted_wqes);

    size_t total_bytes = reg_info.data_sizes[1];
    size_t numel_server = REG_SZ_DATA / sizeof(float);
    // Create a tensor from the raw data (and clone it to own its memory)
    auto updated_tensor = torch::from_blob(srvr_w, { static_cast<long>(numel_server) }, torch::kFloat32).clone();

    w = { updated_tensor };

    // Print the first few updated weight values from server
    {
      std::ostringstream oss;
      oss << "Number of elements in updated tensor: " << updated_tensor.numel() << "\n";
      oss << "Updated weights from server:" << "\n";
      oss << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 10)) << "\n";
      Logger::instance().log(oss.str());
    }

    // Run the training on the updated weights
    std::vector<torch::Tensor> g = runMNISTTrainDummy(w);

    // // Send the updated weights back to the server
    // auto g_flat = torch::cat(g).contiguous();
    // size_t total_bytes_g = g_flat.numel() * sizeof(float);
    // float* raw_ptr_g = g_flat.data_ptr<float>();

    // // 1- copy data to write in your local memory
    // std::memcpy(castV(reg_info.addr_locs[1]), raw_ptr_g, total_bytes_g);

    // // 2- write msg to remote side
    // Logger::instance().log("writing msg ...\n");

    // // Print the first few updated weights sent by client
    // {
    //   std::ostringstream oss;
    //   oss << "Updated weights sent by client:" << "\n";
    //   oss << g_flat.slice(0, 0, std::min<size_t>(g_flat.numel(), 10)) << "\n";
    //   Logger::instance().log(oss.str());
    // }

    // unsigned int total_bytes_g_int = static_cast<unsigned int>(total_bytes_g);
    // (void)norm::write(conn_data, {total_bytes_g_int}, {loc_info}, NetFlags(),
    //                   RemoteInfo(), latency, posted_wqes);

    Logger::instance().log("Client: Done with iteration\n");

  }

  std::cout << "Client done iters: " << i << "\n";

  return 0;
}
