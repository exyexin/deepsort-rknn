from .YOLOv5 import YOLOv5

__all__ = ['build_detector']


def build_detector(cfg, use_cuda):
    return YOLOv5(cfg.YOLOV5.WEIGHT, cfg.YOLOV5.DATA, cfg.YOLOV5.IMGSZ,
                    cfg.YOLOV5.SCORE_THRESH, cfg.YOLOV5.NMS_THRESH, cfg.YOLOV5.MAX_DET, device='cpu')