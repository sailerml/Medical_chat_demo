# Medical_chat_demo
This project is a simple medical Q&A model that involves a direct crosstalk between a traditional expert model and a large language model.
The technical framework diagram of this project is as follow, where the dotted box part is temporarily excluded from the code of this project
![image](https://github.com/sailerml/Medical_chat_demo/assets/10277621/f9c22da4-6691-4f59-bf52-3eef70c3cc02)

## 1. Requirments
   requirments_freeze.txt
## 2. Including models
   
   -  A model for classifying intent to gossip
     | Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
   -  unit test
      ```python
      python ./nlu/sklearn_Classification/clf_model.py
      ```

   
