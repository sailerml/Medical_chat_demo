#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/8 16:55
@file: bert_model.py
@author: baidq
@Software: PyCharm
@desc:
'''

from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
import json

"""
原始库修改

"""

def variable_mapping(num_hidden_layers):
    """映射到官方BERT权重格式
    """
    mapping = {
        'Embedding-Token': ['bert/embeddings/word_embeddings'],
        'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
        'Embedding-Position': ['bert/embeddings/position_embeddings'],
        'Embedding-Norm': [
            'bert/embeddings/LayerNorm/beta',
            'bert/embeddings/LayerNorm/gamma',
        ],
        'Embedding-Mapping': [
            'bert/encoder/embedding_hidden_mapping_in/kernel',
            'bert/encoder/embedding_hidden_mapping_in/bias',
        ],
        'Pooler-Dense': [
            'bert/pooler/dense/kernel',
            'bert/pooler/dense/bias',
        ],
        'NSP-Proba': [
            'cls/seq_relationship/output_weights',
            'cls/seq_relationship/output_bias',
        ],
        'MLM-Dense': [
            'cls/predictions/transform/dense/kernel',
            'cls/predictions/transform/dense/bias',
        ],
        'MLM-Norm': [
            'cls/predictions/transform/LayerNorm/beta',
            'cls/predictions/transform/LayerNorm/gamma',
        ],
        'MLM-Bias': ['cls/predictions/output_bias'],
    }

    for i in range(num_hidden_layers):
        prefix = 'bert/encoder/layer_%d/' % i
        mapping.update({
            'Transformer-%d-MultiHeadSelfAttention' % i: [
                prefix + 'attention/self/query/kernel',
                prefix + 'attention/self/query/bias',
                prefix + 'attention/self/key/kernel',
                prefix + 'attention/self/key/bias',
                prefix + 'attention/self/value/kernel',
                prefix + 'attention/self/value/bias',
                prefix + 'attention/output/dense/kernel',
                prefix + 'attention/output/dense/bias',
            ],
            'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                prefix + 'attention/output/LayerNorm/beta',
                prefix + 'attention/output/LayerNorm/gamma',
            ],
            'Transformer-%d-FeedForward' % i: [
                prefix + 'intermediate/dense/kernel',
                prefix + 'intermediate/dense/bias',
                prefix + 'output/dense/kernel',
                prefix + 'output/dense/bias',
            ],
            'Transformer-%d-FeedForward-Norm' % i: [
                prefix + 'output/LayerNorm/beta',
                prefix + 'output/LayerNorm/gamma',
            ],
        })

    return mapping







def textcnn(inputs, kernel_initializer):
    """
    基于keras实现的textcnn
    :param inputs:
    :param kernel_initializer:
    :return:
    """
    # 3,4,5
    cnn1 = keras.layers.Conv1D(256,
                               3,
                               strides=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_initializer)(inputs) # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)

    cnn2 = keras.layers.Conv1D(256,
                               4,
                               strides=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_initializer)(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

    cnn3 = keras.layers.Conv1D(256,
                               5,
                               strides=1,
                               padding='same',
                               kernel_initializer=kernel_initializer)(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    output = keras.layers.concatenate([cnn1, cnn2, cnn3],axis=-1)
    output = keras.layers.Dropout(0.2)(output)

    return output


def build_bert_model(config_path, checkpoint_path, class_nums):
    """
    构建bert模型用来进行医疗意图的识别
    :param config_path:
    :param checkpoint_path:
    :param class_nums:
    :return:
    """
    # 预加载bert模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )

    # 抽取cls 这个token
    cls_features = keras.layers.Lambda(
        lambda x: x[:,0], # 所有行的第一列
        name='cls-token')(bert.model.output) #shape=[batch_size,768]
    # 抽取所有的token，从第二个到倒数第二个
    all_token_embedding = keras.layers.Lambda(
        lambda x: x[:,1:-1],
        name='all-token')(bert.model.output) #shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(all_token_embedding, bert.initializer) #shape=[batch_size,cnn_output_dim]

    # 特征拼接
    concat_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)

    dense = keras.layers.Dense(units=512,
                               activation='relu',
                               kernel_initializer=bert.initializer)(concat_features)

    output = keras.layers.Dense(units=class_nums,
                                activation='softmax',
                                kernel_initializer=bert.initializer)(dense)

    model = keras.models.Model(bert.model.input, output)
    print(model.summary())

    return model

if __name__ == '__main__':
    config_path = 'D:\\01_data\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'D:\\01_data\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt'
    class_nums = 13
    build_bert_model(config_path, checkpoint_path, class_nums)

