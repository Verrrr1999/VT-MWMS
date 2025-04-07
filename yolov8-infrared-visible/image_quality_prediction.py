import torch
import torch.nn as nn


class ImageQualityPredictionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(ImageQualityPredictionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.silu1 = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.silu2 = nn.SiLU()
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        try:
            x = self.silu1(self.bn1(self.conv1(features)))
            x = self.silu2(self.bn2(self.conv2(x)))
            quality = self.sigmoid(self.conv3(x))
            return quality
        except Exception as e:
            print(f"图像质量预测网络出错: {e}")
            return None


def test_image_quality_prediction_network():
    in_channels = 64
    network = ImageQualityPredictionNetwork(in_channels)
    features = torch.randn(2, in_channels, 32, 32)
    output = network(features)
    if output is not None:
        print(f"图像质量预测网络测试通过，输出形状: {output.shape}")
    else:
        print("图像质量预测网络测试失败")


if __name__ == "__main__":
    test_image_quality_prediction_network()
    