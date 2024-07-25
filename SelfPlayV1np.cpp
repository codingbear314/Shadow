#include <torch/torch.h>
#include <iostream>
#include "ShadowConfigMacro.hpp"
#include "ShadowModel.hpp"  // Assume this is the file containing the model definition
#include "Encode.hpp"

int main() {
    try {
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            std::cerr << "CUDA is not available. This program requires CUDA." << std::endl;
            return 1;
        }
        printf("Cuda is available\n");

        // Create an instance of the model
     
        Shadow_Chess_V1_Resnet model;
        model.to(torch::kCUDA);  // Move the model to the GPU

        // Create a random input tensor
        chess::Board board;
        torch::Tensor input = encode_board(board);
        input = input.unsqueeze(0);  // Add a batch dimension
        input = input.unsqueeze(0);  // Add a channel dimension

        // Move the input tensor to the GPU
        input = input.to(torch::kCUDA);

        // Perform a forward pass
        std::pair<torch::Tensor, torch::Tensor> output = model.forward(input);

        std::cout << "Policy head output size: " << output.first.sizes() << std::endl;
        std::cout << "Value head output size: " << output.second.sizes() << std::endl;

        std::cout << "Policy head output: " << output.first << std::endl;
        std::cout << "Value head output: " << output.second << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: " << e.msg() << std::endl;
        return 1;
    }

    return 0;
}