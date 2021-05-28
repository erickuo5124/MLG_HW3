import numpy as np
from .similarity import cos_sim, pearson_sim

k = 10 # k most similar users

def get_min_index(arr):
  min_index = 0
  for i in range(len(arr)):
    if arr[i] < arr[min_index]:
      min_index = i
  return min_index

def get_ratings(datas, max_value, max_index):
  num_r = datas.shape[1]
  ratings = []
  for i in range(num_r):
    r = np.sum(datas[max_index, i])
    ratings.append(r / k)
  return ratings

def get_value_index_pearson(predict, datas):
  max_value = []
  max_index = []
  for user in predict:
    values = np.zeros(k)
    indexs = np.zeros(k).astype(int)
    for index, data in enumerate(datas):
      sim = pearson_sim(user, data)
      min_index = get_min_index(values)
      if sim > values[min_index]:
        values[min_index] = sim
        indexs[min_index] = index
    max_value.append(values)
    max_index.append(indexs)
  return np.array(max_value), np.array(max_index)

def get_value_index_cos(predict, datas):
  max_value = []
  max_index = []
  for user in predict:
    values = np.zeros(k)
    indexs = np.zeros(k).astype(int)
    for index, data in enumerate(datas):
      sim = cos_sim(user, data)
      min_index = get_min_index(values)
      if sim > values[min_index]:
        values[min_index] = sim
        indexs[min_index] = index
    max_value.append(values)
    max_index.append(indexs)
  return np.array(max_value), np.array(max_index)