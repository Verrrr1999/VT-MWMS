import torch
import torch.nn as nn
import torchvision.ops as ops


class FeatureAlignmentModule(nn.Module):
    def __init__(self, in_channels, deform_groups=1):
        super(FeatureAlignmentModule, self).__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 2 * 3 * 3 * deform_groups, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deform_conv = ops.DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=deform_groups)

    def forward(self, visible_features, infrared_features):
        try:
            assert visible_features.shape[1] == infrared_features.shape[1], "输入特征通道数必须一致"
            assert visible_features.shape[2:] == infrared_features.shape[2:], "输入特征的空间尺寸必须一致"
            concat_features = torch.cat([visible_features, infrared_features], dim=1)
            offset = self.offset_conv(concat_features)
            aligned_infrared_features = self.deform_conv(infrared_features, offset)
            return aligned_infrared_features
        except AssertionError as e:
            print(f"特征对齐模块出错: {e}")
            return None
        except Exception as e:
            print(f"特征对齐模块出现未知错误: {e}")
            return None


def test_feature_alignment_module():
    in_channels = 64
    module = FeatureAlignmentModule(in_channels)
    visible_features = torch.randn(2, in_channels, 32, 32)
    infrared_features = torch.randn(2, in_channels, 32, 32)
    output = module(visible_features, infrared_features)
    if output is not None:
        print(f"特征对齐模块测试通过，输出形状: {output.shape}")
    else:
        print("特征对齐模块测试失败")


if __name__ == "__main__":
    test_feature_alignment_module()
    