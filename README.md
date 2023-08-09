# Medical_chat_demo
This project is a simple medical Q&A model that involves a direct crosstalk between a traditional expert model and a large language model.
The technical framework diagram of this project is as follow, where the dotted box part is temporarily excluded from the code of this project
![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/f9c22da4-6691-4f59-bf52-3eef70c3cc02)

## 1. Requirments
   requirments_freeze.txt
   
## 2. Including models

   - A model for classifying intent to gossip

      |  闲聊意图   | 含义  |
      |  ----  | ----: |
      | greet  | 打招呼 |
      | goodbye  | 再见 |
      | deny  | 否定 |
      | isbot  | 闲聊 |
      | accept  | 接受 |
      | diagnosis  | 医疗问诊 |

      ```python
      python ./nlu/sklearn_Classification/clf_model.py
      ```

   -  A medical Intent Classification Model
      |  医疗意图   | 示例  |  含义  |
      |  ----  | :----:  | ----:  |
      | 定义  | 如何快速识别心绞痛？ | 0 |
      | 病因  | 今天感觉肚子发硬怎么回事？ | 1 |
      | 预防  | 得了肠胃炎多吃什么比较好？ | 2 |
      | 症状  | 咽喉肿痛、剧烈咳嗽、多痰 | 3 |
      | 相关病症  | 药物中毒所致的精神障碍是什么病 | 4 |
      | 治疗方法  | 失眠心悸该怎么办 | 5 |
      | 所属科室  | 成都肛肠医院哪里治疗便秘好? | 6 |
      | 传染性  | 风湿性心脏病会遗传给外孙吗 | 7 |
      | 治愈率  | 变异性哮喘可以治愈吗 | 8 |
      | 禁忌  | 胸壁肿瘤有什么不能吃的？ | 9 |
      | 治疗时间  | 引产后需要休息多长时间？ | 10 |
      | 化验/体检方案  | 心悸要做什么检查 | 11 |
      | 其他  | 冠状动脉加强CT和肺部CT能否一起做？ | 12 |
      
         ```python
         python ./nlu/intent_recg_bert/app.py
         ```
   -  A NER model
      |  实体类型   | 含义  | 实体数量  | 示例  |
      |  ----  | :----:  | :----:  | ----:  |
      | disease  | 病症 | 8807 | 肺炎 |
      | department  | 医疗科目 | 54 | 烧伤科 |
      | drug  | 药品 | 3828 | 布林佐胺滴眼液 |
      | food  | 食物 | 4870 | 草莓 |
      | check  | 诊断检查项目 | 3353 | 支气管造影 |
      | symptom  | 症状 | 5998 | 咳嗽 |
      | producer  | 在售药物 | 17201 | 注射用盐酸氨溴索 |
      | total  | 总计 | 44111 | 约4.4万实体量级 |
      
         ```python
         python ./knowledge_extraction/bilstm/app.py
         ```
## 3. Running
   
   ```python
   python local.py
   ```
## 4. Other resource
    
   chinese_bert: chinese_L-12_H-768_A-12

## 5. TO DO
   -  Answer retrieval and response part
   -  API interface development
   -  Robustness optimization
