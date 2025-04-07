import torch
from infrared_visible_yolov8 import InfraredVisibleYOLOv8
from ultralytics import YOLO


def main():
    try:
        yolov8_backbone = YOLO('yolov8n.pt').model
        model = InfraredVisibleYOLOv8(yolov8_backbone)
        visible_images = torch.randn(2, 3, 640, 640)
        infrared_images = torch.randn(2, 3, 640, 640)
        output = model(visible_images, infrared_images)
        if output is not None:
            print(f"整体测试通过，输出形状: {output.shape}")
        else:
            print("整体测试失败")
    except Exception as e:
        print(f"主程序出错: {e}")


if __name__ == "__main__":
    main()
    