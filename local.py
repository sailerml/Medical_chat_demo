#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2023/8/9
@file: local.py
@author: wwliu
@Software: PyCharm
@desc:
'''

import os
import json
import pdb

from sanic import Sanic, response
from sanic_cors import CORS
from sanic_openapi import swagger_blueprint, doc

from modules import chitchat_bot,medical_bot, classifier
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context

"""
问答流程：
1、用户输入文本
2、对文本进行解析得到语义结构信息
3、根据语义结构去查找知识，返回给用户

文本解析流程：
1、意图识别
    闲聊意图：greet, goodbye, accept, deny, isbot
            greet, goodbye: 需要有回复动作
            accept, deny：需要执行动作
    诊断意图：
            当意图置信度达到一定阈值时(>=0.8)，可以查询该意图下的答案
            当意图置信度较低时(0.4~0.8)，按最高置信度的意图查找答案，反问用户，进行问题澄清
            当意图置信度更低时(<0.4)，拒绝回答
2、槽位填充
    如果输入是一个诊断意图，那么就需要填充语义槽，得到结构化语义
"""

def delete_cache(file_name):
    """
    清除缓存数据，切换账号登入
    :param file_name:
    :return:
    """
    if os.path.exists(file_name):
        os.remove(file_name)



def message(request):
    """
    sanic 入口
    :param request:
    :return:
    """
    # # 获取用户ID和用户输入
    # sender = request.json.get("sender")
    # message = request.json.get("message")

    # local debug
    sender = request["sender"]
    message = request["message"]
    print("sender:{}, message:{}".format(sender, message))
    # 判断用户意图是否属于闲聊类，相当于第一层意图过滤
    user_intent = classifier(message)
    print("user_intent:", user_intent)
    if user_intent in ["greet", "goodbye", "deny", "isbot"]:
        reply = chitchat_bot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(sender)
        print('when accept, all json data is | ', reply)
        reply = reply.get("choice_answer")
        print("01-accept:",reply)
    # diagnosis
    else:
        reply = medical_bot(message, sender)
        if reply["slot_values"]:
            dump_user_dialogue_context(sender,reply)
        reply = reply.get("replay_answer")
    print("reply:",reply)
    return response.json(reply)

if __name__ == '__main__':
    """
    测试用例：
    你是机器人吗
    请问得了心脏病怎么办呢
    心肌炎是什么
    怎么治疗较好呢
    需要治疗多久才会恢复呢？？
    是的
    
    """
    # 打开下面注释可以清除对话日志缓存
    # delete_cache(file_name='s./logs/loginInfo.pkl')


    # local debug
    testjson = {
        "sender":'001',
        "message":'请问得了心脏病怎么办呢'
    }
    response = message(testjson)