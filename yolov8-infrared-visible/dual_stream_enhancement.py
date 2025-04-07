import torch
import torch.nn as nn


class DualStreamAlternatingEnhancementModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(DualStreamAlternatingEnhancementModule, self).__init__()
        self.visible_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.infrared_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, visible_features, infrared_features):
        try:
            assert visible_features.shape[1] == infrared_features.shape[1], "输入特征通道数必须一致"
            assert visible_features.shape[2:] == infrared_features.shape[2:], "输入特征的空间尺寸必须一致"
            visible_attention = self.visible_attention(infrared_features)
            infrared_attention = self.infrared_attention(visible_features)
            enhanced_visible_features = visible_features * visible_attention
            enhanced_infrared_features = infrared_features * infrared_attention
            return enhanced_visible_features, enhanced_infrared_features
        except AssertionError as e:
            print(f"双流交替增强模块出错: {e}")
            return None, None
        except Exception as e:
            print(f"双流交替增强模块出现未知错误: {e}")
            return None, None


def test_dual_stream_enhancement_module():
    in_channels = 64
    module = DualStreamAlternatingEnhancementModule(in_channels)
    visible_features = torch.randn(2, in_channels, 32, 32)
    infrared_features = torch.randn(2, in_channels, 32, 32)
    enhanced_visible, enhanced_infrared = module(visible_features, infrared_features)
    if enhanced_visible is not None and enhanced_infrared is not None:
        print(f"双流交替增强模块测试通过，增强可见特征形状: {enhanced_visible.shape}, 增强红外特征形状: {enhanced_infrared.shape}")
    else:
        print("双流交替增强模块测试失败")


if __name__ == "__main__":
    test_dual_stream_enhancement_module()
    