export USE_RKNN=1
python deepsort.py \
    --VIDEO_PATH ./test.mp4 \
    --config_detection ./configs/yolov5s.yaml \
    --cpu
