#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@file: modules.py
@author: wwliu
@Software: PyCharm
@desc:
'''

import json
import pdb

import requests
import random

from nlu.sklearn_Classification.clf_model import CLFModel
from nlu.intent_recg_bert.app import BertIntentModel
from knowledge_extraction.bilstm.app import MedicalNerModel
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context
from config import *

dict_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/vocab.txt'
config_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:\\01_data\\000_open_source\\00_word-embedding_pretrain-model\chinese_bert\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt'
clf_model = CLFModel('./nlu/sklearn_Classification/model_file/')
BIM = BertIntentModel(dict_path, config_path, checkpoint_path)
NER = MedicalNerModel()

def intent_classifier(text):
    """
    通过post方式请求医疗意图识别分类服务
    基于bert+textcnn实现
    :param text:
    :return:
    """
    res = BIM.predict(text)
    return res


def slot_recognizer(text):
    """
    槽位识别器
    :param text:
    :return:
    """
    res = NER.predict([text])
    return res

def entity_link(mention, etype):
    """
    #TODO 对于识别到的实体mention,如果不是知识库中的标准称谓
    则对其进行实体链指，将其指向一个唯一实体
    :param mention:
    :param etype:
    :return:
    """
    return mention

def classifier(text):
    """
    判断是否是闲聊意图，以及是什么类型闲聊
    :param text:
    :return:
    """
    return clf_model.predict(text)



def semantic_parser(text, user):
    """
    对用户输入文本进行解析,然后填槽，确定回复策略
    :param text:
    :param user:
    :return:
            填充slot_info中的["slot_values"]
            填充slot_info中的["intent_strategy"]
    """
    # 对医疗意图进行二次分类
    intent_receive = intent_classifier(text) # {'confidence': 0.8997645974159241, 'intent': '治疗方法'}
    print("intent_receive:",intent_receive)
    # 实体识别
    slot_receive = slot_recognizer(text)
    print("slot_receive:", slot_receive)

    if intent_receive == -1 or slot_receive == -1 or intent_receive.get("intent")=="其他":
        return semantic_slot.get("unrecognized")

    print("intent:", intent_receive.get("intent"))
    slot_info = semantic_slot.get(intent_receive.get("intent"))
    print("slot_info:", slot_info)
    # 填槽
    slots = slot_info.get("slot_list") # ["Disease"]
    slot_values = {}
    for slot in slots:              # 遍当前意图下的所有槽位,可以设置多个槽位解决任务型问答
        slot_values[slot] = None    # 将槽位置空
        for entities_info in slot_receive:
            print('entities_info | ', entities_info)
            for entity in entities_info["entities"]:
                if slot.lower() == entity["type"]:
                    slot_values[slot] = entity_link(entity["word"], entity["type"])

    last_slot_values = load_user_dialogue_context(user)["slot_values"]  #导入历史对话信息槽值
    for k in slot_values.keys():
        if slot_values[k] is None:
            slot_values[k] = last_slot_values.get(k, None)
    slot_info["slot_values"] = slot_values
    print('final slot values | ', slot_values)

    # 根据意图的置信度来确认回复策略
    # TODO 使用强化学习进行策略选择
    conf = intent_receive.get("confidence")
    if conf >= intent_threshold_config["accept"]:   # >0.8
        slot_info["intent_strategy"] = "accept"
    elif conf >= intent_threshold_config["deny"]:   # >0.4
        slot_info["intent_strategy"] = "clarify"
    else:
        slot_info["intent_strategy"] = "deny"

    print("semantic_parser:",slot_info)
    return slot_info


def get_answer(slot_info):
    """
    根据不同的回复策略，可利用指令学习去大语言模型中查询答案
    :param slot_info:
    :return: 在slot_info中增加"replay_answer"这一项
    """
    cql_template = slot_info.get("cql_template")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")

    if not slot_values:
        return slot_info

    if strategy == "accept":

        slot_info["replay_answer"] = "when accept answer"

    elif strategy == "clarify":
        # 0.4 < 意图置信度 < 0.8时，进行问题澄清
        pattern = ask_template.format(**slot_values)
        print("pattern for clarity:", pattern)

        slot_info["replay_answer"] = "when clarify answer"


    elif strategy == "deny":
        slot_info["replay_answer"] = slot_info.get("deny_response")

    print("get_answer:", slot_info)
    return slot_info

def chitchat_bot(intent):
    """
    如果是闲聊，就从闲聊的回复语料里随机选择一个返回给用户
    :param intent:
    :return:
    """
    return random.choice(chitchat_corpus.get(intent))

def medical_bot(text, user):
    """
    如果确定是诊断意图，则使用该函数进行诊断问答
    :param text:
    :param user:
    :return:
    """
    semantic_slot = semantic_parser(text, user)
    answer = get_answer(semantic_slot)
    return answer




