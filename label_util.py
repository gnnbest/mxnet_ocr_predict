# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import csv

import pickle
import os.path as osp
import logging
import numpy


curr_path = osp.dirname(osp.abspath(__file__))


class LabelUtil:
    _log = logging

    # dataPath
    def __init__(self, fname=None):
        if fname is None:
            self.load_unicode_set()
        else:
            self.load_unicode_set(fname)


    def load_unicode_set(self, vocab_fname=curr_path+'/data/vocab.pkl'):
        self.unicodeFilePath = vocab_fname
        with open(vocab_fname, 'rb') as f:
            d = pickle.load(f)
        self.byChar = d['r_vocab']
        self.byIndex = d['vocab']
        self.size = len(self.byChar)
        print("vocab size:{}".format(self.size))


    def idx2onehot(self, idx):
        ''' 序号转换为 one_hot  np
        '''
        a = np.zeros((self.size,))
        a[idx] = 1
        return a


    def word2onehot(self, w):
        return self.idx2onehot(self.byChar[w])


    def onehot2idx(self, o):
        return np.where(o==1)[0][0]

    
    def w2i(self, w):
        if w not in self.byChar:
            return 1    # "unknown"
        return self.byChar[w]


    def i2w(self, idx):
        return self.byIndex[idx]

    
    def word2num(self, s):
        return [self.w2i(c) for c in s]

    def num2word(self, n):
        return [self.i2w(c) for c in n]

    def to_unicode(self, src, index):
        # 1 byte
        code1 = int(ord(src[index + 0]))

        index += 1

        result = code1

        return result, index

    def convert_word_to_grapheme(self, label):

        result = []

        index = 0
        while index < len(label):
            (code, nextIndex) = self.to_unicode(label, index)

            result.append(label[index])

            index = nextIndex

        return result, "".join(result)

    def convert_word_to_num(self, word):

        try:
            label_list, _ = self.convert_word_to_grapheme(word)

            label_num = []

            for char in label_list:
                # skip word
                if char == "":
                    pass
                else:
                    if char not in self.byChar:
                        char = '~'
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)

    def convert_bi_graphemes_to_num(self, word):
            label_num = []

            for char in word:
                # skip word
                if char == "":
                    pass
                else:
                    label_num.append(int(self.byChar[char]))

            # tuple typecast: read only, faster
            return tuple(label_num)


    def convert_num_to_word(self, num_list):
        try:
            label_list = []
            for num in num_list:
                label_list.append(self.byIndex[num])
            return ''.join(label_list)

        except AttributeError:
            self._log.error("unicodeSet is not loaded")
            exit(-1)

        except KeyError as err:
            self._log.error("unicodeSet Key not found: %s" % err)
            exit(-1)





vocab = LabelUtil()
