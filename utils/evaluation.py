import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def RMSE(predict, answer):
  return sqrt(mean_squared_error(answer, predict))

def Recall(predict, answer):
  predict = np.array(predict)
  answer = np.array(answer)
  user_num = predict.shape[0]
  recall_sum = 0
  for i in range(user_num):
    avg = np.average(answer[i])
    
    # remove no rating situation
    if avg == 0: 
      user_num -= 1
      continue
    
    hit = 0
    rel = 0
    for p, a in zip(predict[i], answer[i]):
      if a >= avg:
        rel += 1
        if p >= avg:
          hit += 1
    recall_sum += hit / rel

  # consider divided by zero
  return recall_sum / user_num if user_num else 1
    
    

def NDCG(predict, answer):
  predict = np.array(predict).astype(float)
  answer = np.array(answer).astype(int)
  user_num = predict.shape[0]
  NDCG_sum = 0

  for i in range(user_num):
    IDCG = 0
    for index, rel in enumerate(np.sort(answer[i])[::-1]):
      IDCG += rel / np.log2(index+2) # 0 -> -Inf, 1 -> 0.
      if rel == 0:
        break
    if IDCG == 0:
      user_num -= 1
      continue
    
    dtype = [('predict', float), ('rel', int)]
    p = [(v, r) for v, r in zip(predict[i], answer[i])]
    p = np.array(p, dtype=dtype)
    DCG = 0
    for index, rel in enumerate(np.sort(p, order='predict')[::-1]):
      DCG += rel[1] / np.log2(index+2)

    NDCG_sum += DCG / IDCG

  return NDCG_sum / user_num if user_num else 1

# input lists of rating
def print_evalution(predict, answer):
  print(f'RMSE: {RMSE(predict, answer)}')
  print(f'Recall: {Recall(predict, answer)}')
  print(f'NDCG: {NDCG(predict, answer)}')