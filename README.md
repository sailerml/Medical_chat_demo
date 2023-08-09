# Medical_chat_demo
This project is a simple medical Q&A model that involves a direct crosstalk between a traditional expert model and a large language model.
The technical framework diagram of this project is as follow, where the dotted box part is temporarily excluded from the code of this project
![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/f9c22da4-6691-4f59-bf52-3eef70c3cc02)

## 1. Requirments
   requirments_freeze.txt
   
## 2. Including models

   - A model for classifying intent to gossip
      ![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/35fd19e9-771b-4c5f-bb1e-9dd664850d77)
     
    
     | Tables   |      Are      |  Cool |
      |----------|:-------------:|------:|
      | col 1 is |  left-aligned | $1600 |
      | col 2 is |    centered   |   $12 |
      | col 3 is | right-aligned |    $1 |
         ```python
         python ./nlu/sklearn_Classification/clf_model.py
         ```
   -  A medical Intent Classification Model
      ![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/e7351ba0-b532-47b1-b2f1-c7e580d41bde)
      

      
         ```python
         python ./nlu/intent_recg_bert/app.py
         ```
   -  A NER model
      ![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/51a1e24d-c188-4759-99a4-4e0a033e9a24)

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
