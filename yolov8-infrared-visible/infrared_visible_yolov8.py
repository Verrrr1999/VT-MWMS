import torch
import torch.nn as nn
from ultralytics import YOLO
from feature_alignment import FeatureAlignmentModule
from region_image_quality_fusion import RegionImageQualityGuidedFusionModule
from dual_stream_enhancement import DualStreamAlternatingEnhancementModule
from cross_modal_interaction import CrossModalFeatureInteractionModule


class InfraredVisibleYOLOv8(nn.Module):
    def __init__(self, yolov8_backbone, in_channels=256):
        super(InfraredVisibleYOLOv8, self).__init__()
        self.visible_backbone = yolov8_backbone
        self.infrared_backbone = yolov8_backbone
        self.feature_alignment = FeatureAlignmentModule(in_channels)
        self.fusion_module = RegionImageQualityGuidedFusionModule(in_channels)
        self.dae_module = DualStreamAlternatingEnhancementModule(in_channels)
        self.cmfi_module = CrossModalFeatureInteractionModule(in_channels)

    def forward(self, visible_images, infrared_images):
        try:
            visible_features = self.visible_backbone(visible_images)
            infrared_features = self.infrared_backbone(infrared_images)
            aligned_infrared_features = self.feature_alignment(visible_features, infrared_features)
            if aligned_infrared_features is None:
                return None
            fused_features = self.fusion_module(visible_features, aligned_infrared_features)
            if fused_features is None:
                return None
            enhanced_visible, enhanced_infrared = self.dae_module(visible_features, aligned_infrared_features)
            if enhanced_visible is None or enhanced_infrared is None:
                return None
            output = self.cmfi_module(enhanced_visible, enhanced_infrared)
            return output
        except Exception as e:
            print(f"红外 - 可见光跨模态 YOLOv8 模型出错: {e}")
            return None


def test_infrared_visible_yolov8():
    yolov8_backbone = YOLO('yolov8n.pt').model
    model = InfraredVisibleYOLOv8(yolov8_backbone)
    visible_images = torch.randn(2, 3, 640, 640)
    infrared_images = torch.randn(2, 3, 640, 640)
    output = model(visible_images, infrared_images)
    if output is not None:
        print(f"红外 - 可见光跨模态 YOLOv8 模型测试通过，输出形状: {output.shape}")
    else:
        print("红外 - 可见光跨模态 YOLOv8 模型测试失败")


if __name__ == "__main__":
    test_infrared_visible_yolov8()
    