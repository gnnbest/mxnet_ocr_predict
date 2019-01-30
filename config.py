#!/bin/env python3
#coding: utf-8


# 存储动态配置信息

class Config:
    batch_size = 8
    epoch_num = 30
    
    rnn_layers_num = 2
    rnn_hidden_num = 256

    vocab_size = -1
    vocab = None
    r_vocab = None

    max_img_width = 800    # 图像最大宽度
    img_height = 36        # 图像目标高度
    img_channels = 3
    max_label_len = 60     # 最大 label 长度
    
    bucket_num = 10      # 桶的数目，等分

    learning_rate = 0.00005

    train_show_step = 2000
    eval_show_step = 100

    embed_size = 100   # 字向量长度
    seq2seq_rnn_layer_num = 3
    seq2seq_rnn_hidden_num = 256

    target_vocab_size = -1  # 校正后的字典长度
    target_vocab = None
    target_r_vocab = None
