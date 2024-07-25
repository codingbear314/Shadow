#pragma once
// Shadow C++ / ShadowModel.hpp
// Part of the Shadow C++ chess engine project.
// 2024, by js314

#include <torch/torch.h>

class ResidualBlock : public torch::nn::Module {
public:
    static const int expansion = 4;

    ResidualBlock(int64_t in_channel, int64_t out_channel, int64_t stride = 1)
        : conv_bottleneck1(register_module("conv_bottleneck1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, 1).bias(false)))),
        bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channel))),
        conv_main(register_module("conv_main",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channel, out_channel, 3)
                .stride(stride).padding(1).bias(false)))),
        bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channel))),
        conv_bottleneck2(register_module("conv_bottleneck2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channel, out_channel* expansion, 1).bias(false)))),
        bn3(register_module("bn3", torch::nn::BatchNorm2d(out_channel* expansion)))
    {
        if (stride != 1 || in_channel != out_channel * expansion) {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel * expansion, 1)
                    .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channel * expansion)
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

private:
    torch::nn::Conv2d conv_bottleneck1 = nullptr, conv_main = nullptr, conv_bottleneck2 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr, bn2 = nullptr, bn3 = nullptr;
    torch::nn::Sequential shortcut = nullptr;
};

class Shadow_Chess_V1_Resnet : public torch::nn::Module {
public:
    Shadow_Chess_V1_Resnet(int64_t num_classes = 4672)
        : conv3d(register_module("conv3d",
            torch::nn::Conv3d(torch::nn::Conv3dOptions(1, 64, { 3, 3, 6 }).stride(1).padding({ 1, 1, 0 }).bias(false)))),
        bn3d(register_module("bn3d", torch::nn::BatchNorm3d(64))),
        layer1(make_layer(64, 3)),
        layer2(make_layer(128, 4, 2)),
        layer3(make_layer(256, 6, 2)),
        layer4(make_layer(512, 3, 2)),
        avgpool(register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })))),
        policyHead(register_module("policyHead", torch::nn::Sequential(
            torch::nn::Linear(512 * ResidualBlock::expansion, 1024),
            torch::nn::ReLU(true),
            torch::nn::Linear(1024, num_classes)
        ))),
        valueHead(register_module("valueHead", torch::nn::Sequential(
            torch::nn::Linear(512 * ResidualBlock::expansion, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 1),
            torch::nn::Tanh()
        )))
    {}

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Input shape: (batch_size, 1, 8, 8, 6)
        // Or: (1, 8, 8, 6) which gets unsqueezed to (1, 1, 8, 8, 6)
        if (x.sizes() == torch::IntArrayRef({ 1, 8, 8, 6 })) {
            x = x.unsqueeze(0);
        }

        // After conv3d: (batch_size, 64, 8, 8, 1)
        x = torch::relu(bn3d(conv3d(x)));

        // After squeeze: (batch_size, 64, 8, 8)
        x = x.squeeze(4);  // Remove the last dimension

        // After layer1: (batch_size, 256, 8, 8)
        x = layer1->forward(x);

        // After layer2: (batch_size, 512, 4, 4)
        x = layer2->forward(x);

        // After layer3: (batch_size, 1024, 2, 2)
        x = layer3->forward(x);

        // After layer4: (batch_size, 2048, 1, 1)
        x = layer4->forward(x);

        // After avgpool: (batch_size, 2048, 1, 1)
        x = avgpool(x);

        // After flatten: (batch_size, 2048)
        x = x.flatten(1);

        // Policy output shape: (batch_size, 4672)
        auto policy = policyHead->forward(x);

        // Value output shape: (batch_size, 1)
        auto value = valueHead->forward(x);

        return { policy, value };
    }

private:
    torch::nn::Sequential make_layer(int64_t out_channel, int64_t num_blocks, int64_t stride = 1) {
        torch::nn::Sequential layers;
        layers->push_back(ResidualBlock(in_channel, out_channel, stride));
        in_channel = out_channel * ResidualBlock::expansion;
        for (int i = 1; i < num_blocks; ++i) {
            layers->push_back(ResidualBlock(in_channel, out_channel));
        }
        return layers;
    }

    torch::nn::Conv3d conv3d = nullptr;
    torch::nn::BatchNorm3d bn3d = nullptr;
    torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
    torch::nn::AdaptiveAvgPool2d avgpool = nullptr;
    torch::nn::Sequential policyHead = nullptr, valueHead = nullptr;
    int64_t in_channel = 64;
};