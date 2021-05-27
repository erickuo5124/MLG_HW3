import numpy as np

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