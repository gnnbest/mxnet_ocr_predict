#!/bin/env python
#coding: utf-8

import mxnet as mx
import numpy as np
import os.path as osp
import random
import cv2
import logging
from config import Config
import math
from label_util import LabelUtil


class Batch:
    def __init__(self, data, label, bucket_key):
        self.data = data
        self.label = label
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]

    @property
    def provide_label(self):
        if self.label:
            return [('label', self.label[0].shape)]
        else:
            return None


class DataSource:
    def __init__ (self, path, batch_size=32, img_pp=False):
        idx_fname = osp.splitext(path)[0] + '.idx'
        self.db_ = mx.recordio.MXIndexedRecordIO(idx_fname, path, 'r')
        self.db_.open()
        self.img_preprocess_ = img_pp
        self.num_ = len(self.db_.keys)
        self.batch_size_ = batch_size
        self.base_off_ = 0
        #self.ops_cnt = [0,0,0,0,0]
        
    def __del__(self):
        self.db_.close()

    def __iter__(self):
        ''' 因为图片本来已经打乱，所以按照顺序走吧 :)
        '''
        cnt = (self.num_ - self.base_off_) // self.batch_size_
        self.db_.seek(self.base_off_)
        
        for i in range(cnt):
            imgs, labels, bucket_key = self.load_imgs_labels(i*self.batch_size_ + self.base_off_)
            yield Batch(imgs, labels, bucket_key)

        self.base_off_ += 1
        self.base_off_ %= self.batch_size_
        
        raise StopIteration


    def extract_picture(self, imgid, fname=None):
#        curr = self.db_.tell()
        self.db_.seek(imgid)
        raw_data = self.db_.read()
        data = mx.recordio.unpack(raw_data)
        label = data[0].label
        img = cv2.imdecode(np.frombuffer(data[1], dtype=np.uint8), 1)
        cv2.imwrite(fname, img)
#        self.db_.seek(curr)


    @property
    def count(self):
        return self.num_

    def _blur(self, img):
        kernel_size = (random.randint(1,3), random.randint(1,3))
        return cv2.blur(img, kernel_size)
        
    def _grad(self, img):
        kernel_size = (random.randint(1,3), random.randint(1,3))
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_size)

    def _erode(self, img):
        kernel_size = (random.randint(1,3), random.randint(1,3))
        return cv2.erode(img, kernel_size, random.randint(1,3))

    def _dilate(self, img):
        kernel_size = (random.randint(1,3), random.randint(1,3))
        return cv2.dilate(img, kernel_size, random.randint(1,3))

    def _nothing(self, img):
        return img

    def preprocess_img(self, img0):
        ''' 随机处理一下图片，比如模糊局部，锐化局部 ...    增加 nothing 的可能性
        '''
        ops = [self._dilate, self._nothing, self._grad, self._blur, self._nothing, self._erode, self._nothing]
        ops_name = ["dilate", "nothing", "grad", "blur", "nothing", "erode", "nothing",]
        op = random.randint(0, len(ops)-1)
        #self.ops_cnt[op] += 1
        
        img = ops[op](img0)

        # if self.ops_cnt[op] % 1000 == 0:
        #     cv2.imwrite("{}_orig.jpg".format(ops_name[op]), img0)
        #     cv2.imwrite("{}.jpg".format(ops_name[op]), img)

        return img


    def load_imgs_labels(self, base_idx):
        ''' 加载 base_idx 到 base_idx+batch_size 个 img，根据 shape[2] 做成统一的 ...
            
            XXX: 为了使用 bucketing 模式训练，应该设计一组"阶梯" ...
        '''
        imgs = []
        labels = []

        max_width = 0

        for i in range(base_idx, base_idx + self.batch_size_):
            raw_data = self.db_.read()
            data = mx.recordio.unpack(raw_data)

            label = data[0].label # label: numpy, shape=(yyy,)
            if isinstance(label, float):
                label = np.array([label], dtype=np.int)
            else:
                label = label.astype(np.int)

            color = 1
            if Config.img_channels == 1:
                color = 0
            img = cv2.imdecode(np.frombuffer(data[1], dtype=np.uint8), color)
            # 保持比例拉伸为 (36, x)
            h,w = img.shape[:2]
            tw = int(Config.img_height * w / h)
            th = Config.img_height
            img = cv2.resize(img, (tw,th))

            if img.shape[1] >= Config.max_img_width:
                img = img[:,:Config.max_img_width,:]

            if self.img_preprocess_:
                img = self.preprocess_img(img)

            # img.shape = (Config.img_height, xxx, 3)
            img = np.swapaxes(img, 0, 2)    # (3, xxx, 60)
            img = np.swapaxes(img, 1, 2)    # (3, 60, xxx)

            img = img.astype(dtype=np.float32)

            img /= 255.0
            img -= 0.5

            if img.shape[-1] > max_width:
                max_width = img.shape[-1]

            imgs.append(img)
            labels.append(label)

        # 检查 max_width 落在那个 bucket 区间
        n = int(math.ceil(1.0 * Config.max_img_width / Config.bucket_num)) # 每个区间的 width 
        max_width = (max_width + n - 1) // n * n
        bucket_key = max_width // n - 1

        # 将 img 转化为 (3, 60, bucket_cc)
        imgs2 = []
        for img in imgs:
            pad_n = max_width - img.shape[-1]
            if pad_n:
                img = np.pad(img, ((0,0),(0,0),(0,pad_n)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            img = img.reshape(Config.img_channels*Config.img_height*max_width)
            imgs2.append(img)
            
        imgs = np.vstack(imgs2)
        imgs = imgs.reshape((self.batch_size_, Config.img_channels, Config.img_height, max_width))

        ll = int(math.ceil(1.0*Config.max_label_len/Config.bucket_num))
        label_len = ll * (bucket_key+1)    # 此处 img_key 相当于 bucket_key
        #label_len = Config.max_label_len

        labels2 = []
        for label in labels:
            pad_n = label_len - label.shape[0]
            if pad_n < 0:
                # print('bucket key={}, img width={}'.format(bucket_key, max_width))
                # print('???? label_len={}, label: {}, {}'.format(label_len, label, _vocab.convert_num_to_word(label.tolist())))
                label = label[:label_len]
            elif pad_n:
                label = np.pad(label, (0,pad_n), 'constant', constant_values=(0, 0)) # 0: <pad>
            label = label.reshape((1,-1))
            labels2.append(label)

        labels = np.vstack(labels2)
        labels = labels.reshape((self.batch_size_, label_len))

        # 返回之前，转化为 mx.nd.array
        return [mx.nd.array(imgs)], [mx.nd.array(labels)], bucket_key




''' 句子，支持 bucket
'''
class DataSourceSentence:
    class Batch:
        def __init__(self, data_names, data, label_names, label, bucket_key):
            self.data = data
            self.label = label
            self.data_names = data_names
            self.label_names = label_names
            self.bucket_key = bucket_key

            self.pad = 0
            self.index = None  # TODO: what is index?

        @property
        def provide_data(self):
            return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

        @property
        def provide_label(self):
            return [(n, x.shape) for n, x in zip(self.label_names, self.label)]



    def __init__(self, fname, src_voc, tar_voc, batch_size=128, ctx=mx.cpu()):
        self.batch_size_ = batch_size
        blen = Config.max_label_len // Config.bucket_num
        self.bucket_sizes_ = [blen*(i+1) for i in range(Config.bucket_num)] #
        self.bucket_sizes_[-1] = Config.max_label_len
        self.bucket_data_ = [[] for i in range(Config.bucket_num)]
        self.src_vocab_ = src_voc
        self.tar_vocab_ = tar_voc
        self.ctx_ = ctx
        self.bucket_iternums_ = self.load_from_file(fname, self.bucket_data_)


    @property
    def provide_data(self):
        return [("source", (self.batch_size_, Config.max_label_len)), ("target", (self.batch_size_, Config.max_label_len))]

    @property
    def provide_label(self):
        return [("label", (self.batch_size_, Config.max_label_len))]


    def load_from_file(self, fname, buckets):
        ''' 将数据加载到 self.bucket_data_ 中，如果某个bucket数目不够 batch_size，则合并到下一个 bucket 中
            返回每个 bucket 应该迭代的次数 ...
        '''
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if len(line) < 2:
                    continue
                for i in range(Config.bucket_num):
                    if len(line) < self.bucket_sizes_[i]:
                        buckets[i].append(line)
                # 超长已经扔掉了

        for i in range(Config.bucket_num-1):
            if len(buckets[i]) < self.batch_size_:
                buckets[i+1] += buckets[i]
                buckets[i] = []
        
        bucket_iternums = []
        for i in range(Config.bucket_num):
            bucket_iternums.append(len(buckets[i])//self.batch_size_)
        return bucket_iternums


    def __iter__(self):
        for i in range(Config.bucket_num):
            logging.info("#%d bucket: size=%d, num=%d"%(i,self.bucket_sizes_[i],self.bucket_iternums_[i]))
            

        for i in range(Config.bucket_num):
            curr_bucket_size, curr_bucket_data, curr_buck_iternum = self.bucket_sizes_[i], self.bucket_data_[i], self.bucket_iternums_[i]
            for j in range(curr_buck_iternum):
                samples = random.sample(curr_bucket_data, self.batch_size_)
                source = mx.nd.zeros(shape=(self.batch_size_, self.bucket_sizes_[i]),ctx=self.ctx_)
                target = mx.nd.zeros_like(source,ctx=self.ctx_)
                label = mx.nd.zeros_like(source, ctx=self.ctx_)
                for n,s in enumerate(samples):
                    sids = self.src_vocab_.word2num(s)
                    tids = self.tar_vocab_.word2num(["<s>"] + list(s))
                    lids = self.tar_vocab_.word2num(list(s) + ["</s>"])
                    
                    source[n,:len(sids)] = mx.nd.array(sids)
                    target[n,:len(tids)] = mx.nd.array(tids)
                    label[n,:len(lids)] = mx.nd.array(lids)
                yield self.Batch(["source", "target"], [source, target], ["label"], [label], i)
        raise StopIteration


    def reset(self):
        pass



if __name__ == '__main__':
    curr_path = osp.dirname(osp.abspath(__file__))
    ds = DataSourceSentence(curr_path+"/data/samples.txt", curr_path+"/data/vocab.pkl", curr_path+"/data/target_vocab.pkl")
    for batch in ds:
        print(batch)
        break
