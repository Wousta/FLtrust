#include "../../RcConn/include/rc_conn.hpp"
#include "../../rdma-api/include/rdma-api.hpp"
#include "../../shared/util.hpp"
#include "include/mnistTrain.hpp"
#include "include/globalConstants.hpp"

#include <logger.hpp>
#include <lyra/lyra.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define MSG_SZ 32
using ltncyVec = std::vector<std::pair<int, std::chrono::nanoseconds::rep>>;

int exec_rdma_op(int loc_info_idx, int rem_info_idx, uint64_t size, int op_type) {
  return 0;
}

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

  // addr
  Logger::instance().log("Client: id = " + std::to_string(id) + "\n");
  Logger::instance().log("Client: srvr_ip = " + srvr_ip + "\n");
  Logger::instance().log("Client: port = " + port + "\n");
  addr_info.ipv4_addr = strdup(srvr_ip.c_str());
  addr_info.port = strdup(port.c_str());

  // Data structures for server and this client
  int srvr_ready_flag = 0;
  float* srvr_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));
  int clnt_ready_flag = 0;
  float* clnt_w = reinterpret_cast<float*> (malloc(REG_SZ_DATA));

  // memory registration
  reg_info.addr_locs.push_back(castI(&srvr_ready_flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.addr_locs.push_back(castI(&clnt_ready_flag));
  reg_info.addr_locs.push_back(castI(srvr_w));
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.data_sizes.push_back(MIN_SZ);
  reg_info.data_sizes.push_back(REG_SZ_DATA);
  reg_info.permissions = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  // connect to server
  Logger::instance().log("Connecting to server\n");
  RcConn conn;
  int ret = conn.connect(addr_info, reg_info);
  comm_info conn_data = conn.getConnData();

  std::vector<torch::Tensor> w;
  for (int round = 1; round <= GLOBAL_ITERS; round++) {

    LocalInfo flag_info;
    flag_info.indices.push_back(SRVR_READY_IDX);
    do {
      norm::read(conn_data, { sizeof(int) }, { flag_info }, NetFlags(),
        RemoteInfo(), latency, posted_wqes);
    } while (srvr_ready_flag != SRVR_READ_READY);

    std::cout << "Client read flag = " << srvr_ready_flag << "\n";

    // Read the weights from the server
    LocalInfo local_srvr_data;
    local_srvr_data.indices.push_back(SRVR_W_IDX);
    RemoteInfo remote_srvr_info;
    remote_srvr_info.indx = SRVR_W_IDX;
    (void)norm::read(conn_data, { REG_SZ_DATA }, { local_srvr_data }, NetFlags(),
                    remote_srvr_info, latency, posted_wqes);

    size_t numel_server = REG_SZ_DATA / sizeof(float);
    auto updated_tensor = torch::from_blob(srvr_w, { static_cast<long>(numel_server) }, torch::kFloat32).clone();
    w = { updated_tensor };

    // Print the first few updated weight values from server
    {
      std::ostringstream oss;
      oss << "Number of elements in updated tensor: " << updated_tensor.numel() << "\n";
      oss << "Updated weights from server:" << "\n";
      oss << w[0].slice(0, 0, std::min<size_t>(w[0].numel(), 12)) << "\n";
      Logger::instance().log(oss.str());
    }

    // Run the training on the updated weights
    std::vector<torch::Tensor> g = runMNISTTrainDummy(w);

    // Send the updated weights back to the server
    auto g_flat = torch::cat(g).contiguous();
    size_t total_bytes_g = g_flat.numel() * sizeof(float);
    float* raw_ptr_g = g_flat.data_ptr<float>();
    std::memcpy(castV(reg_info.addr_locs[CLNT_W_IDX]), raw_ptr_g, total_bytes_g);

    LocalInfo local_clnt_data;
    local_clnt_data.indices.push_back(CLNT_W_IDX);
    RemoteInfo remote_clnt_info;
    remote_clnt_info.indx = CLNT_W_IDX;
    unsigned int total_bytes_g_int = static_cast<unsigned int>(total_bytes_g);
    (void)norm::write(conn_data, {total_bytes_g_int}, {local_clnt_data}, NetFlags(),
                      remote_clnt_info, latency, posted_wqes);

    // Print the first few updated weights sent by client
    {
      std::ostringstream oss;
      oss << "Updated weights sent by client:" << "\n";
      oss << g_flat.slice(0, 0, std::min<size_t>(g_flat.numel(), 10)) << "\n";
      Logger::instance().log(oss.str());
    }

    // Update the ready flag
    clnt_ready_flag = CLNT_READ_READY;
    LocalInfo local_clnt_flag;
    local_clnt_flag.indices.push_back(CLNT_READY_IDX);
    RemoteInfo remote_clnt_flag;
    remote_clnt_flag.indx = CLNT_READY_IDX;
    (void)norm::write(conn_data, {sizeof(int)}, {local_clnt_flag}, NetFlags(),
                      remote_clnt_flag, latency, posted_wqes);

    Logger::instance().log("Client: Done with iteration\n");

  }

  std::cout << "Client done\n";

  return 0;
}
