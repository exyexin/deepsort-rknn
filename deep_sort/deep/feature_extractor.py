import onnxruntime as ort
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import os

from .model import Net
from .resnet import resnet18
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultTrainer
# from fastreid.utils.checkpoint import Checkpointer

from rknnlite.api import RKNNLite as RKNN
USE_RKNN = True
print(f'==========USE_RKNN:{USE_RKNN}===========')

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        
        # self.net = Net(reid=True)
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        # self.net.load_state_dict(state_dict if 'net_dict' not in state_dict else state_dict['net_dict'], strict=False)
        # self.net.to(self.device)

        # 加载 ONNX 模型
        # self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider'])
        
        self.extractor = RKNN(verbose=True)
        ret = self.extractor.load_rknn(model_path)
        if ret != 0:
            print('load model failed!')
            exit(1)

        ret = self.extractor.init_runtime(core_mask=RKNN.NPU_CORE_1)
        if ret != 0:
            print('rknn runtime init failed!')
            exit(1) 


        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # onnx_model_path = model_path.replace(".t7", ".onnx")
        # # 创建一个示例输入张量
        # example_input = torch.randn(1, 3, 128, 64).to(self.device)  # 根据模型输入调整张量大小
        # # 将模型转换为 ONNX 格式并保存
        # torch.onnx.export(self.net, example_input, onnx_model_path, export_params=True, opset_version=11,
        #               do_constant_folding=True, input_names=['input'], output_names=['output'],
        #               dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            if USE_RKNN:
                print('USE_RKNN _resize-->')
                img = cv2.resize(im.astype(np.float32), size)
            else:
                img = cv2.resize(im.astype(np.float32) / 255., size)
            return img

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops).cpu().numpy()
        # with torch.no_grad():
        #     im_batch = im_batch.to(self.device)
        #     features = self.net(im_batch)
        
        # 进行推理
        # input_name = self.session.get_inputs()[0].name
        # output_name = self.session.get_outputs()[0].name
        # features = self.session.run([output_name], {input_name: im_batch})[0]
        
        features = []
        batch_size = len(im_batch)
        
        # Run the model for each sample in the batch individually
        for i in range(batch_size):
            single_batch = im_batch[i:i+1]  # Create a batch of size 1
            # feature = self.session.run([output_name], {input_name: single_batch})[0]
            feature = self.extractor.inference([single_batch])[0]
            features.append(feature)
            breakpoint()
        
        # cat features
        features = np.concatenate(features, axis=0)
        
        return features


class FastReIDExtractor(object):
    def __init__(self, model_config, model_path, use_cuda=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_config)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.net = DefaultTrainer.build_model(cfg)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        Checkpointer(self.net).load(model_path)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        height, width = cfg.INPUT.SIZE_TEST
        self.size = (width, height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
