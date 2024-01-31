#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 16:10
@file: app.py
@author: baidq
@Software: PyCharm
@desc:
'''
import tensorflow as tf


from gevent import pywsgi
from bert4keras.tokenizers import Tokenizer

from keras.backend.tensorflow_backend import set_session
from nlu.intent_recg_bert.bert_model import build_bert_model
#from tensorflow.python.keras.backend import set_session

class BertIntentModel(object):
    """
    基于bert实现的医疗意图识别模型
    """
    def __init__(self, dict_path, config_path, checkpoint_path):
        super(BertIntentModel, self).__init__()


        self.dict_path = dict_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        self.label_list  = [line.strip() for line in open(r"D:\00_code\001_git_store\medical_chat_demo\nlu\intent_recg_bert\label", "r", encoding="utf8")]
        #import pdb; pdb.set_trace()
        self.id2label = {idx:label for idx, label in enumerate(self.label_list)}

        self.tokenizer = Tokenizer(self.dict_path)
        self.model = build_bert_model(self.config_path, self.checkpoint_path, 13)
        self.model.load_weights(r"D:\00_code\001_git_store\medical_chat_demo\nlu\intent_recg_bert\model\best_model_weights")

    def predict(self, text):
        """
        对用户输入的单条文本进行预测
        :param text:
        :return:
        """
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=60)
        predict = self.model.predict([[token_ids], [segment_ids]])
        rst = {l:p for l,p in zip(self.label_list, predict[0])}
        rst = sorted(rst.items(), key= lambda kv:kv[1], reverse=True)
        print(rst[0])
        intent, confidence = rst[0]
        return {"intent": intent, "confidence":float(confidence)}


config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

# 医疗意图识别分类器
dict_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/vocab.txt'
config_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt'

BIM= BertIntentModel(dict_path, config_path, checkpoint_path)

if __name__ == '__main__':
    r = BIM.predict("淋球菌性尿道炎的症状")
    print(r)

