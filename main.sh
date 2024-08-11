export USE_RKNN=1
sudo /home/user/.conda/envs/yolov7-rknn/bin/python deepsort.py \
    --VIDEO_PATH ./test.mp4 \
    --config_detection ./configs/yolov5s.yaml \
    --cpu