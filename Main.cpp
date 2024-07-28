# define SDW_USE_CUDA

#include <torch/torch.h>
#include "ShadowModel.hpp"
#include "Encode.hpp"
#include <iostream>


void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " - Shape: " << tensor.sizes()
        << ", Device: " << tensor.device()
        << ", Type: " << tensor.dtype() << std::endl;
}

int main() {
    std::cout << "Shadow C++ alpha 0.1 by js314" << std::endl;

    try {
        // Check CUDA availability
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available");
        }
        std::cout << "CUDA is available" << std::endl;

        // Set random seed for reproducibility
        torch::manual_seed(42);

        // Create an instance of the model
        auto model = std::make_shared<Shadow_Chess_V1_Resnet>();

        // Explicitly move each parameter of the model to CUDA
        for (auto& param : model->parameters()) {
            param = param.to(torch::kCUDA);
        }
        model->to(torch::kCUDA);
        model->to_cuda();

        // Verify model parameters are on CUDA
        for (const auto& param : model->parameters()) {
            if (param.device().type() != torch::kCUDA) {
                throw std::runtime_error("Model parameter not on CUDA");
            }
        }

        // Put the model in evaluation mode
        model->eval();

        // Create input tensor
        chess::Board bd;
        auto input = encode_board(bd);
        input = input.unsqueeze(0).unsqueeze(0).to(torch::kCUDA);

        // Print input tensor info
        print_tensor_info(input, "Input");

        // Print model's first parameter info
        print_tensor_info(model->parameters().front(), "First model parameter");

        // Disable gradient computation for inference
        torch::NoGradGuard no_grad;

        std::cout << "Model loaded successfully!" << std::endl;

        // Forward pass
        auto [policy, value] = model->forward(input);

        // Print output info
        print_tensor_info(policy, "Policy");
        print_tensor_info(value, "Value");

        std::cout << "Policy output (first 10 elements): " << policy.slice(1, 0, 10) << std::endl;
        std::cout << "Value output: " << value.item<float>() << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}