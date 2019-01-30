#!/bin/env python
#coding: utf-8


import mxnet as mx
import math
from config import Config



ww = math.ceil(1.0 * Config.max_img_width / Config.bucket_num)
ll = math.ceil(1.0 * Config.max_label_len / Config.bucket_num)




def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut


def res_pass(data, ylen, xlen):
    ''' 使用N组
    '''
    filter_list = [32, 64, 96, 128, 256]
    pool_list = [1, 0, 0, 0, 0]
    drop_list = [0.3, 0.2, 0.1, 0, 0]

    assert(len(filter_list) == len(pool_list))
    assert(len(drop_list) == len(pool_list))

    for i,f in enumerate(filter_list):
        data = residual_unit(data, f, (1,1), False, "res_%d"%i)

        if pool_list[i]:
            data = mx.sym.Pooling(data, kernel=(3,3), stride=(2,2), pool_type="max")
            ylen = int(math.floor((ylen - 3) / 2 + 1))
            xlen = int(math.floor((xlen - 3) / 2 + 1))

        if drop_list[i] > 0.0:
            data = mx.sym.Dropout(data, p=drop_list[i])

    return data, ylen, xlen


def cnn_pass(data, axis2, axis3):
    ker = (3,3)
    stride = (1,1)
    pad = (2,2)
    data = mx.sym.Convolution(data, name='conv1', num_filter=64, kernel=ker, stride=stride, pad=pad)
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1] + 2*pad[1]) / stride[1] + 1))  # (simg - kernel) / stride + 1
    axis2 = int(math.floor((axis2 - ker[0] + 2*pad[0]) / stride[0] + 1))

    # pool_stride = (2,2)
    # pool_ker = (2,2)
    # pool_pad = (0,0)
    # data = mx.sym.Pooling(data, pool_type='max', pad=pool_pad, kernel=pool_ker, stride=pool_stride)    
    # axis3 = int(math.floor((axis3 - pool_ker[1] + 2*pool_pad[1]) / pool_stride[1] + 1))
    # axis2 = int(math.floor((axis2 - pool_ker[0] + 2*pool_pad[0]) / pool_stride[0] + 1))

    ker = (3,3)
    stride = (1,1)
    pad = (0,0)
    data = mx.sym.Convolution(data, name='conv2', num_filter=128, kernel=ker, stride=stride, pad=pad)
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1] + 2*pad[1]) / stride[1] + 1))
    axis2 = int(math.floor((axis2 - ker[0] + 2*pad[0]) / stride[0] + 1))

    pool_stride = (2,2)
    pool_ker = (2,2)
    pool_pad = (0,0)
    data = mx.sym.Pooling(data, pool_type='max', pad=pool_pad, kernel=pool_ker, stride=pool_stride)    
    axis3 = int(math.floor((axis3 - pool_ker[1] + 2*pool_pad[1]) / pool_stride[1] + 1))
    axis2 = int(math.floor((axis2 - pool_ker[0] + 2*pool_pad[0]) / pool_stride[0] + 1))

    ker = (3,3)
    stride = (2,2)
    pad = (0,0)
    data = mx.sym.Convolution(data, name='conv3', num_filter=256, kernel=ker, stride=stride, pad=pad)
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1] + 2*pad[1]) / stride[1] + 1))
    axis2 = int(math.floor((axis2 - ker[0] + 2*pad[0]) / stride[0] + 1))

    data = mx.sym.Dropout(data, p=0.2)

    return data, axis2, axis3


def build_net(bucket_key, for_train, using_resnet=False):
    ''' 构造网络，返回 (sym, data_names, label_names)

        label 长度根据 bucket_key 决定：

    '''

    # simg 输入图像宽度
    simg = (bucket_key+1) * ww
    axis3 = simg
    axis2 = Config.img_height
    
    # label 长度
    slab = int((bucket_key+1) * ll)
    #slab = Config.max_label_len

    data = mx.sym.var(name='data')      # data shape: (batch_size, 3, 60, simg)
    if for_train:
        label = mx.sym.var(name='label')    # label shape: (batch_size, slab)
    else:
        label = None

    if using_resnet:
        data,axis2,axis3 = res_pass(data, axis2, axis3)
    else:
        data,axis2,axis3 = res_pass(data, axis2, axis3)

    # 将 data[3] 轴完全分割 ...
    slice_cnt = axis3
    data = mx.sym.split(data, num_outputs=slice_cnt, axis=3)

    data = [ d for d in data ]

    # rnn
    stack = mx.rnn.SequentialRNNCell()
    for i in range(Config.rnn_layers_num):
        cell = mx.rnn.GRUCell(Config.rnn_hidden_num, prefix='gru_{}'.format(i))
        stack.add(cell)
        #stack.add(mx.rnn.DropoutCell(0.3, prefix='drop_{}'.format(i)))

    outputs, states = stack.unroll(slice_cnt, data)

    # fc: 使用共享参数，将 outputs[n] 映射到 vocab 空间
    #cls_weight = mx.sym.var(name='cls_weight')
    #cls_bias = mx.sym.var(name='cls_bias')
    
    # seq = []
    # for i,o in enumerate(outputs):
    #     fc = mx.sym.FullyConnected(data=o, name='fco_{}'.format(i), 
    #             num_hidden=Config.vocab_size, weight=cls_weight, bias=cls_bias)
    #     fc = mx.sym.Dropout(fc, p=0.3)
    #     seq.append(fc)
    # data = mx.sym.concat(*seq, dim=0)

    data = mx.sym.concat(*outputs, dim=0)

    if Config.vocab_size == -1:
        Config.vocab_size = 6118 # XXX:
        
    pred = mx.sym.FullyConnected(data, num_hidden=Config.vocab_size, name='fc')

#    print('build_net: bucket_key={}, img_width={}, slice_cnt={}, label_length={}'.format(bucket_key, 
#            simg, slice_cnt, slab))

    if for_train:
        # warpctc 要把 batch 合并呢？
        label = mx.sym.Reshape(data=label, shape=(-1,))
        label = mx.sym.Cast(data=label, dtype='int32')
        out = mx.sym.WarpCTC(data=pred, label=label, input_length=slice_cnt, label_length=slab)
        return (out, ['data'], ['label'])
    else:
        out = mx.sym.softmax(data=pred, name='sm')
        return (out, ['data'], None)



def build_train_net(bucket_key):
    return build_net(bucket_key, True)


def build_train_net_resnet(bucket_key):
    return build_net(bucket_key, True, using_resnet=True)


def build_infer_net(bucket_key):
    return build_net(bucket_key, False)


def build_lm_net(bucket_key, train):
    ''' 构造一个 seq2seq 模型，希望能调整 ocr 输出的文本中的错别字
        input seq 中可能错别字，漏字，多字 ...在 ocr 中典型的错误是“相近字”，

        test 1: 对 input seq 进行几层 conv，再进行 deconv 输出 output seq
            
    '''
    data = mx.sym.var(name="data")      # shape=(batch_size, 60)    # input seq, padding to 60
    if train:
        label = mx.sym.var(name="label")   # shape=(batch_size, 60)    # output seq, padding to 60

    wordvec = mx.sym.var(name="wordvec_embed_weight")
    data = mx.sym.Embedding(data, name="embed", weight=wordvec, input_dim=Config.vocab_size, output_dim=Config.embed_size)   # output (batch, vocab_size, embed_size)

    data = mx.sym.reshape(data, shape=(0,1,Config.embed_size,-1))   # (batch_size, 1, embed_size, 60)
    data = mx.sym.swapaxes(data, 2, 3)                              # (batch_size, 1, 60, embed_size)

    def conv(data, kernel, pad, num_filter, act_type="relu", stride=(1,1), drop=0.0):
        convx = mx.sym.Convolution(data, kernel=kernel, pad=pad, num_filter=num_filter, stride=stride)
        actx = mx.sym.Activation(convx, act_type=act_type)
        if drop > 0.0000001:
            actx = mx.sym.Dropout(actx, p=drop)
        return actx

    def deconv(data, kernel, pad, num_filter, act_type="relu", stride=(1,1), drop=0.0):
        deconvx = mx.sym.Deconvolution(data, kernel=kernel, pad=pad, num_filter=num_filter, stride=stride)
        actx = mx.sym.Activation(deconvx, act_type=act_type)
        if drop > 0.0000001:
            actx = mx.sym.Dropout(actx, p=drop)
        return actx

    num_filter_1 = 128
    num_filter_2 = 256

    ### conv
    conv_1_list = [conv(data, (i+1, 1), (1,0), num_filter_1) for i in range(6)]
    conv_2_list = [conv(conv_1_list[i], (i+1, 1), (0,0), num_filter_2) for i in range(6)]

    ### deconv
    deconv2_list = [deconv(conv_2_list[i], (i+1, 1), (0,0), num_filter_2) for i in range(6)]
    deconv1_list = [deconv(deconv2_list[i], (i+1, 1), (1,0), num_filter_1) for i in range(6)]

    sum = mx.sym.concat(*deconv1_list, dim=1) 
    mean = mx.sym.mean(sum, axis=1)   # (batch, 1, 1, 60)
    
    pred = mx.sym.squeeze(mean)     # (batch, 60)

    if train:
        label_embedded = mx.sym.Embedding(data=label, name="embed_label", weight=wordvec, input_dim=Config.vocab_size, output_dim=Config.embed_size)
        loss = mx.sym.sqrt(mx.sym.pow((pred - label_embedded), 2))
        loss = mx.sym.MakeLoss(loss)
        group = mx.sym.Group([mx.sym.BlockGrad(pred), loss])
        return group
    else:
        return pred
    

def build_lm_train_net(bucket_key):
    return build_lm_net(bucket_key, train=True)


def build_lm_infer_net(bucket_key):
    return build_lm_net(bucket_key, train=False)



def build_seq2seq(bucket_key, train, src_voc_size, tar_voc_size):
    ''' 
    '''
    seq_len = (bucket_key+1) * ll

    source = mx.sym.var(name="source")
    target = mx.sym.var(name="target")
    if train:
        label = mx.sym.var(name="label")

    # encoder
    enc_embed = mx.sym.Embedding(source, input_dim=src_voc_size, output_dim=100)
    enc_wordvec = mx.sym.split(enc_embed, seq_len, squeeze_axis=1)   # [batch, 100] * seq_len
    enc_wordvec_list = [v for v in enc_wordvec]

    enc_stack = mx.rnn.SequentialRNNCell()
    for i in range(Config.seq2seq_rnn_layer_num):
        enc_stack.add(mx.rnn.GRUCell(num_hidden=Config.seq2seq_rnn_hidden_num, prefix="enc_l%d"%i))
    enc_outputs,enc_states = enc_stack.unroll(inputs=enc_wordvec_list, length=seq_len)
    print("seq length=%d"%seq_len)
    
    # state transfer
    states = mx.sym.concat(*enc_states)
    states_trans = mx.sym.FullyConnected(states, num_hidden=Config.seq2seq_rnn_layer_num*Config.seq2seq_rnn_hidden_num) # num_hidden 应为 decoder 的层数...
    states_trans = mx.sym.Activation(states_trans, act_type="tanh")

    # decoder
    dec_embed = mx.sym.Embedding(target, input_dim=tar_voc_size, output_dim=100)
    dec_wordvec = mx.sym.split(dec_embed, seq_len, squeeze_axis=1)
    dec_wordvec_list = [v for v in dec_wordvec]

    dec_stack = mx.rnn.SequentialRNNCell()
    for i in range(Config.seq2seq_rnn_layer_num):
        dec_stack.add(mx.rnn.GRUCell(num_hidden=Config.seq2seq_rnn_hidden_num, prefix="dec_l%d"%i))

    # 使用 state transfer 作为 decoder 的 init state
    states = mx.sym.split(states_trans, num_outputs=Config.seq2seq_rnn_layer_num)
    states = [s for s in states]
    dec_outputs, dec_states = dec_stack.unroll(inputs=dec_wordvec_list, length=seq_len, begin_state=states)

    dec_output = mx.sym.concat(*dec_outputs, dim=0)    # (batch * seq_len, Config.seq2seq_rnn_hidden_num)
    pred = mx.sym.FullyConnected(dec_output, num_hidden=tar_voc_size)     # (batch * seq_len, tar_voc_size)

    if train:
        label = mx.sym.reshape(label, shape=(-1,))      # (batch * seq_len)
        sm = mx.sym.SoftmaxOutput(pred, label)          # (batch * seq_len, tar_voc_size)
    else:
        sm = mx.sym.softmax(pred)


    return sm, ["source", "target"], ["label"]



def build_seq2seq_train(bucket_key):
    return build_seq2seq(bucket_key, True, Config.vocab_size, Config.target_vocab_size)

def build_seq2seq_infer(bucket_key):
    return build_seq2seq(bucket_key, False, Config.vocab_size, Config.target_vocab_size)


if __name__ == '__main__':
    net = build_net(9)
    print(net)
