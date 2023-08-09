# Medical_chat_demo
This project is a simple medical Q&A model that involves a direct crosstalk between a traditional expert model and a large language model.
The technical framework diagram of this project is as follow, where the dotted box part is temporarily excluded from the code of this project
![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/f9c22da4-6691-4f59-bf52-3eef70c3cc02)

## 1. Requirments
   requirments_freeze.txt
## 2. Including models
   -  A model for classifying intent to gossip
      ![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/b73e82c7-38c6-4ae7-a7a4-12234e6e225f)
   -  unit test
      ```python
      python ./nlu/sklearn_Classification/clf_model.py
      ```

   
