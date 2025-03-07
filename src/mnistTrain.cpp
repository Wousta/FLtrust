#include "include/mnistTrain.hpp"

#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int runMNISTTrain() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
  return 1;
}