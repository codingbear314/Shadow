#pragma once
// Shadow C++ / ShadowModel.hpp
// Part of the Shadow C++ chess engine project.
// 2024, by js314

#include <torch/torch.h>

class ResidualBlock : public torch::nn::Module {
public:
    static const int expansion = 4;

    ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
        : conv_bottleneck1(register_module("conv_bottleneck1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)))),
        bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channels))),
        conv_main(register_module("conv_main",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                .stride(stride).padding(1).bias(false)))),
        bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channels))),
        conv_bottleneck2(register_module("conv_bottleneck2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels* expansion, 1).bias(false)))),
        bn3(register_module("bn3", torch::nn::BatchNorm2d(out_channels* expansion)))
    {
        if (stride != 1 || in_channels != out_channels * expansion) {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels * expansion, 1)
                    .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels * expansion)
            ));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1(conv_bottleneck1(x)));
        out = torch::relu(bn2(conv_main(out)));
        out = bn3(conv_bottleneck2(out));
        out += shortcut->is_empty() ? x : shortcut->forward(x);
        return torch::relu(out);
    }

    void to_cuda() {
#ifdef SDW_USE_CUDA
        conv_bottleneck1->to(torch::kCUDA);
        bn1->to(torch::kCUDA);
        conv_main->to(torch::kCUDA);
        bn2->to(torch::kCUDA);
        conv_bottleneck2->to(torch::kCUDA);
        bn3->to(torch::kCUDA);
        if (!shortcut->is_empty()) {
            shortcut->to(torch::kCUDA);
        }
#endif
    }

private:
    torch::nn::Conv2d conv_bottleneck1, conv_main, conv_bottleneck2;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
    torch::nn::Sequential shortcut;
};

class Shadow_Chess_V1_Resnet : public torch::nn::Module {
public:
    Shadow_Chess_V1_Resnet(int64_t num_classes = 4672)
        : conv3d(register_module("conv3d",
            torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 64, { 3, 3, 6 }).stride(1).padding({ 1, 1, 0 }).bias(false)))),
        bn3d(register_module("bn3d", torch::nn::BatchNorm3d(64))),
        layer1(make_layer(64, 64, 2)),
        layer2(make_layer(64 * ResidualBlock::expansion, 128, 3, 2)),
        layer3(make_layer(128 * ResidualBlock::expansion, 256, 4, 2)),
        layer4(make_layer(256 * ResidualBlock::expansion, 512, 2, 1)),
        avgpool(register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })))),
        policyHead(register_module("policyHead", torch::nn::Sequential(
            torch::nn::Linear(512 * ResidualBlock::expansion, 2048),
            torch::nn::ReLU(true),
            torch::nn::Linear(2048, num_classes)
        ))),
        valueHead(register_module("valueHead", torch::nn::Sequential(
            torch::nn::Linear(512 * ResidualBlock::expansion, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 1),
            torch::nn::Tanh()
        )))
    {}

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        //std::cout << "!" << std::endl;
        // Input shape: (batch_size, 1, 8, 8, 6)
        // Or: (1, 8, 8, 6) which gets unsqueezed to (1, 1, 8, 8, 6)
        if (x.dim() == 4) {
            x = x.unsqueeze(0);
        }

        //std::cout << "Apply conv3d and bn3d" << std::endl;
        // After conv3d: (batch_size, 64, 8, 8, 1)
        x = torch::relu(bn3d(conv3d(x)));

        //std::cout << "Squeeze the last dimension" << std::endl;
        // After squeeze: (batch_size, 64, 8, 8)
        x = x.squeeze(4);  // Remove the last dimension
        // Print the shape
        //std::cout << "After squeeze: " << x.sizes() << std::endl;

        // After layer1: (batch_size, 256, 8, 8)
        //std::cout << "Apply layer1" << std::endl;
        x = layer1->forward(x);
        //std::cout << "After layer1: " << x.sizes() << std::endl;

        // After layer2: (batch_size, 512, 4, 4)
        //std::cout << "Apply layer2" << std::endl;
        x = layer2->forward(x);
        //std::cout << "After layer2: " << x.sizes() << std::endl;

        // After layer3: (batch_size, 1024, 2, 2)
        //std::cout << "Apply layer3" << std::endl;
        x = layer3->forward(x);
        //std::cout << "After layer3: " << x.sizes() << std::endl;

        // After layer4: (batch_size, 2048, 1, 1)
        //std::cout << "Apply layer4" << std::endl;
        x = layer4->forward(x);
        //std::cout << "After layer4: " << x.sizes() << std::endl;

        // After avgpool: (batch_size, 2048, 1, 1)
        //std::cout << "Apply avgpool" << std::endl;
        x = avgpool(x);
        //std::cout << "After avgpool: " << x.sizes() << std::endl;

        // After flatten: (batch_size, 2048)
        //std::cout << "Flatten the tensor" << std::endl;
        x = x.flatten(1);
        //std::cout << "After flatten: " << x.sizes() << std::endl;

        // Policy output shape: (batch_size, 4672)
        auto policy = policyHead->forward(x);

        // Value output shape: (batch_size, 1)
        auto value = valueHead->forward(x);

        return { policy, value };
    }

    void to_cuda() {
#ifdef SDW_USE_CUDA
		conv3d->to(torch::kCUDA);
		bn3d->to(torch::kCUDA);
		avgpool->to(torch::kCUDA);
		policyHead->to(torch::kCUDA);
		valueHead->to(torch::kCUDA);
#endif
	}

private:
    torch::nn::Sequential make_layer(int64_t in_channels, int64_t out_channels, int64_t num_blocks, int64_t stride = 1) {
        torch::nn::Sequential layers;
        auto block = std::make_shared<ResidualBlock>(in_channels, out_channels, stride);
        block->to_cuda();
        layers->push_back(block);
        for (int i = 1; i < num_blocks; ++i) {
            block = std::make_shared<ResidualBlock>(out_channels * ResidualBlock::expansion, out_channels);
            block->to_cuda();
            layers->push_back(block);
        }
        return layers;
    }

    torch::nn::Conv3d conv3d;
    torch::nn::BatchNorm3d bn3d;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::AdaptiveAvgPool2d avgpool;
    torch::nn::Sequential policyHead, valueHead;
};