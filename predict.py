#!/bin/env python3
#coding: utf-8

import mxnet as mx
from network import build_infer_net
from config import Config
import cv2
import numpy as np
from train import Batch
import sys
import os.path as osp
from label_util import LabelUtil
import math



''' 构造 Bucketing model 用于 ocr 识别,
    输入图像, 将保持比例拉伸为 (60, xxx), 放到对应的 bucket 中 ...
'''
import os.path as osp
curr_path = osp.dirname(osp.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

#curr_path = osp.dirname(osp.abspath(__file__))
epoch = 1
prefix = curr_path + '/ocr'
sys.path.insert(0, prefix)

img_height = 36     # 拉伸后图像高度


class OCRPredictor:
    def __init__(self, gpu_idx, default_bucket_key=Config.bucket_num-1):
        self._vocab = LabelUtil()
        self.gpu_idx = gpu_idx
        print('******gpu: ', gpu_idx)
        self._mod = self.load_model(default_bucket_key)

    def pred(self, img, txt=True, merge=True):
        batch = self.prepare_img_batch(img)
        self._mod.forward(batch, is_train=False)
        return self.get_result(txt, merge)

    def prepare_img_batch(self, img):
        ''' 将图片保持比例拉伸为 height=60, 然后填充到对应的 bucket 大小, 作为网络输入
        '''
        assert(img.shape[2] == 3) # BGR
        aspect = 1.0 * img.shape[0] / img.shape[1]
        x = int(img_height * img.shape[1] / img.shape[0])
        size = (x, img_height)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 其实换不换无所谓 ...

        lab_len = int(math.ceil(1.0*Config.max_label_len / Config.bucket_num))
        seg_len = int(math.ceil(1.0*Config.max_img_width / Config.bucket_num))

        seg = img.shape[1] // seg_len + 1
        width = seg_len * seg
        if x < width:
            # 需要补齐
            pad = np.zeros((img_height, width-x, 3), dtype=np.uint8)
            img = np.hstack((img, pad))

        assert(img.shape == (img_height,width,3))
        img = np.swapaxes(img, 0, 2)    # (ch, c, r)
        img = np.swapaxes(img, 2, 1)    # (ch, r, c)

        img = img.astype(dtype=np.float32)
        img /= 255.0
        img -= 0.5

        img = img.reshape((1,3,img_height,width))
        return Batch(data=[mx.nd.array(img)], label=None, bucket_key=seg-1)

    def load_model(self, key):
        gpu_idx = self.gpu_idx

        len_per_seg = int(math.ceil(1.0*Config.max_img_width / Config.bucket_num))
        self._mod = mx.mod.BucketingModule(sym_gen=build_infer_net, default_bucket_key=key, context=mx.gpu(gpu_idx))
        self._mod.bind(data_shapes=[('data',(1,3,img_height,(key+1)*len_per_seg))], 
                label_shapes=None,
                for_training=False)
        
        # load checkpoint
        _,args,auxs = mx.model.load_checkpoint(prefix, epoch)
        self._mod.set_params(args, auxs)
        return self._mod

    def get_result(self, txt, merge):
        ''' 获取 ctc 的输出, 然后合并相邻相同的输出 '''
        out = self._mod.get_outputs()[0]
        # out.shape = (slice, vocab_size)
        idxs = np.argmax(out, axis=1)
        idxs = idxs.asnumpy().astype(dtype=np.int32).tolist()
        if merge:
            nums = self.merge(idxs)
        else:
            nums = idxs
        if txt:
            return self._vocab.convert_num_to_word(nums)
        else:
            return nums

    def merge(self, nums):
        ''' 将 nums 中, 相邻并且相等的合并
        '''
        merged = []
        curr = -1
        for n in nums:
            # if n == 0:
            #     continue
            if n != curr:
                merged.append(n)
                curr = n
        # 删除 merged 中的所有 0
        return [ x for x in merged if x != 0 ]


if __name__ == '__main__':
    pred = OCRPredictor(0)
    img = cv2.imread(sys.argv[1])
    print(pred.pred(img, txt=True, merge=True))
