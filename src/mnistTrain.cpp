#include "include/mnistTrain.hpp"
#include "include/logger.hpp"

//#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int counter = 1;

// The model instance
Net model;

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 1;

// The batch size for testing.
const int64_t kTestBatchSize = 1;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 1;

std::vector<torch::Tensor> runMNISTTrainDummy() {
  std::cout << "Running dummy MNIST training\n";
  // static counter to update the tensor value each time the function is called
  static int dummy_value = 0;
  dummy_value++;

  std::vector<torch::Tensor> w;
  // Create 10 small tensors which update based on dummy_value.
  for (int i = 0; i < 10; i++) {
    // Each tensor here holds one float; it will be different each call.
    w.push_back(torch::tensor({static_cast<float>(dummy_value + i)}, torch::kFloat32));
  }
  return w;
}

template <typename DataLoader>
void train(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     /*weight=*/{},
                     torch::Reduction::Sum)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  // Logger::instance().log(
  //     "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
  //     test_loss,
  //     static_cast<double>(correct) / dataset_size);
  Logger::instance().log("Testing donete\n");
}

std::vector<torch::Tensor> runMNISTTrain() {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    Logger::instance().log("CUDA available! Training on GPU.\n");
    device_type = torch::kCUDA;
  } else {
    Logger::instance().log("Training on CPU.\n");
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  //Net model;
  model.to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    std::printf("now we test\n");
    test(model, device, *test_loader, test_dataset_size);
  }

  // Flatten and concatenate all parameters into one contiguous tensor.
  std::vector<torch::Tensor> flat_params;
  for (const auto& param : model.parameters()) {
      // Flatten the parameter tensor to 1-D.
      flat_params.push_back(param.view(-1));
  }

  // Return the model parameters
  
  return flat_params;
}