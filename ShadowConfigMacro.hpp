#pragma once

#include <torch/torch.h>

namespace Shadow_Config {
	torch::Device device(
		(torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU
	); // If the device is set to GPU, every tensor will be on GPU by default
}