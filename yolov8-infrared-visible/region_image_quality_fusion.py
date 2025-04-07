import torch
import torch.nn as nn
from image_quality_prediction import ImageQualityPredictionNetwork


class RegionImageQualityGuidedFusionModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(RegionImageQualityGuidedFusionModule, self).__init__()
        self.visible_quality_network = ImageQualityPredictionNetwork(in_channels, hidden_channels)
        self.infrared_quality_network = ImageQualityPredictionNetwork(in_channels, hidden_channels)

    def forward(self, visible_features, infrared_features):
        try:
            assert visible_features.shape[1] == infrared_features.shape[1], "输入特征通道数必须一致"
            assert visible_features.shape[2:] == infrared_features.shape[2:], "输入特征的空间尺寸必须一致"
            visible_quality = self.visible_quality_network(visible_features)
            infrared_quality = self.infrared_quality_network(infrared_features)
            if visible_quality is None or infrared_quality is None:
                return None
            fused_features = visible_features * visible_quality + infrared_features * infrared_quality
            return fused_features
        except AssertionError as e:
            print(f"区域图像质量引导融合模块出错: {e}")
            return None
        except Exception as e:
            print(f"区域图像质量引导融合模块出现未知错误: {e}")
            return None


def test_region_image_quality_fusion_module():
    in_channels = 64
    module = RegionImageQualityGuidedFusionModule(in_channels)
    visible_features = torch.randn(2, in_channels, 32, 32)
    infrared_features = torch.randn(2, in_channels, 32, 32)
    output = module(visible_features, infrared_features)
    if output is not None:
        print(f"区域图像质量引导融合模块测试通过，输出形状: {output.shape}")
    else:
        print("区域图像质量引导融合模块测试失败")


if __name__ == "__main__":
    test_region_image_quality_fusion_module()
    