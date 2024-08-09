# Installation

```
conda create -n deepsort python=3.7

conda activate deepsort

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple requirements.txt
```

# Inference

```
python deepsort_simple.py \
    --VIDEO_PATH ./test.mp4 \
    --config_detection ./configs/yolov5s.yaml
```

The results will be stored in the `./outputs` directory.
