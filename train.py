#!/bin/env python3
#coding: utf-8


''' 使用 bucket model 训练
'''


import mxnet as mx
import numpy as np
import os
import os.path as osp
import argparse
import sys
import cv2
from collections import namedtuple
from network import build_train_net, build_train_net_resnet
import math
import pickle
from config import Config
from stt_metric import STTMetric
import random
from data_source import Batch, DataSource


ctx = mx.gpu(1)
curr_path = osp.dirname(osp.abspath(__file__))
prefix = 'ocr'


def load_vocab(fname=curr_path+'/data/vocab.pkl'):
    with open(fname, 'rb') as f:
        d = pickle.load(f)

    return d['vocab'], d['r_vocab']


Config.vocab, Config.r_vocab = load_vocab()
Config.vocab_size = len(Config.vocab)

from label_util import LabelUtil

_vocab = LabelUtil()


def save_checkpoint(mod, prefix, epoch):
    sym,_,_, = mod._sym_gen(Config.bucket_num-1)
    sym.save('{}-symbol.json'.format(prefix))
    mod.save_params(prefix + '-%04d.params' % epoch)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, help='resume epoch')
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("train_set", help="the train set in ./data/")
    parser.add_argument("--resnet", action="store_true")
    args = parser.parse_args()

    global prefix
    if args.resnet:
        prefix = prefix+"-rn"
    
    batch_size = Config.batch_size
    epochs = Config.epoch_num

    loss_metric = STTMetric(batch_size=batch_size, is_epoch_end=False)
    eval_metric = STTMetric(batch_size=batch_size, is_epoch_end=True)

    train_set = args.train_set

    ds_train = DataSource(curr_path+'/data/{}.rec'.format(train_set), batch_size, img_pp=args.blur)
    ds_test = DataSource(curr_path+'/data/test.rec', batch_size)

    print("==== load train set: {}".format(curr_path+"/data/{}.rec".format(train_set)))

    if args.resnet:
        print("using RESNet")
        mod = mx.mod.BucketingModule(build_train_net_resnet, default_bucket_key=Config.bucket_num-1, context=ctx)
    else:
        print("using CNN")
        mod = mx.mod.BucketingModule(build_train_net, default_bucket_key=Config.bucket_num-1, context=ctx)

    # bind 时，使用 default_bucket_key 计算
    ww = int(math.ceil(1.0 * Config.max_img_width / Config.bucket_num))

    data_shapes = [('data', (batch_size, Config.img_channels, Config.img_height, ww * Config.bucket_num))]
    label_shapes = [('label', (batch_size, Config.max_label_len))]
    mod.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=True)

    if args.resume:
        from_epoch = args.resume
        print('RESUME from {}'.format(from_epoch))
        sym,args,auxs = mx.model.load_checkpoint(prefix, from_epoch)
        mod.set_params(args, auxs)
    else:
        from_epoch = -1
        mod.init_params(mx.init.Xavier(factor_type='in', magnitude=2.34)) 
        
#    mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': Config.learning_rate, 'momentum': 0.9,
#            "lr_scheduler": mx.lr_scheduler.FactorScheduler(step=60000, factor=0.9)})
    mod.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': Config.learning_rate})
    
    # go
    for e in range(from_epoch+1, epochs):
#        save_checkpoint(mod, prefix, e)

        # test
        eval_metric.reset()
        for i,batch in enumerate(ds_test):
            mod.forward(batch, is_train=False)
            if i % Config.eval_show_step == Config.eval_show_step -1:
                mod.update_metric(eval_metric, batch.label)


        # train
        loss_metric.reset()
        for i,batch in enumerate(ds_train):
            mod.forward_backward(batch)
            mod.update()

            if i % Config.train_show_step == Config.train_show_step - 1:
                print('==> epoch:{}, batch:{}'.format(e, i))
                mod.update_metric(loss_metric, batch.label)

            if i % 100000 == 99999:
                save_checkpoint(mod, prefix, e)

        # save checkpoint
        save_checkpoint(mod, prefix, e)

        # 根据 e 调整 lr
#        if (e+1) % Config.lr_reduce_epoch:
#            pass


if __name__ == '__main__':
    def test_ds():
        ds = DataSource(osp.dirname(osp.abspath(__file__))+'/data/train.rec', batch_size=32)
        for batch in ds:
            print(batch.provide_data, batch.provide_label, batch.bucket_key)
#    test_ds()
    train()
