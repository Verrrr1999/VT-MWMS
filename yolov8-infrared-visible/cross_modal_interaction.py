import torch
import torch.nn as nn


class CrossModalFeatureInteractionModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(CrossModalFeatureInteractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, visible_features, infrared_features):
        try:
            assert visible_features.shape[1] == infrared_features.shape[1], "输入特征通道数必须一致"
            assert visible_features.shape[2:] == infrared_features.shape[2:], "输入特征的空间尺寸必须一致"
            combined_features = torch.cat([visible_features, infrared_features], dim=1)
            x = self.relu(self.conv1(combined_features))
            residual = self.conv2(x)
            output = visible_features + infrared_features + residual
            return output
        except AssertionError as e:
            print(f"跨模态特征交互模块出错: {e}")
            return None
        except Exception as e:
            print(f"跨模态特征交互模块出现未知错误: {e}")
            return None


def test_cross_modal_feature_interaction_module():
    in_channels = 64
    module = CrossModalFeatureInteractionModule(in_channels)
    visible_features = torch.randn(2, in_channels, 32, 32)
    infrared_features = torch.randn(2, in_channels, 32, 32)
    output = module(visible_features, infrared_features)
    if output is not None:
        print(f"跨模态特征交互模块测试通过，输出形状: {output.shape}")
    else:
        print("跨模态特征交互模块测试失败")


if __name__ == "__main__":
    test_cross_modal_feature_interaction_module()
    