# User-based Collaborative Filtering with consine similarity
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from utils.evaluation import print_evalution

# parameter
path = './datas/processed/movie/user_movie.pkl'
predict_num = 100 # number of predict users
test_size = 0.2
k = 10 # k most similar users

# functions
def cos_sim(a, b):
  return dot(a, b)/(norm(a)*norm(b))

def get_min_index(arr):
  min_index = 0
  for i in range(len(arr)):
    if arr[i] < arr[min_index]:
      min_index = i
  return min_index

def get_value_index(predict, datas):
  max_value = []
  max_index = []
  for user in predict:
    values = np.zeros(k)
    indexs = np.zeros(k)
    for index, data in enumerate(datas):
      sim = cos_sim(user, data)
      min_index = get_min_index(values)
      if sim > values[min_index]:
        values[min_index] = sim
        indexs[min_index] = index
    max_value.append(values)
    max_index.append(indexs)

  return np.array(max_value), np.array(max_index)

def get_ratings(datas, max_value, max_index):
  num_r = datas.shape[1]
  ratings = []
  for i in range(num_r):
    r = np.sum(datas[:, i])
    ratings.append(r / k)

  return ratings
  

if __name__ == '__main__':
  df = pd.read_pickle(path)

  datas = df.to_numpy().astype('int')

  user_num = datas.shape[0]
  rnd = np.random.randint(user_num, size=predict_num)
  predict = datas[rnd]
  datas = np.delete(datas, rnd, 0)

  test_length = int(datas.shape[1] * test_size)
  train_length = datas.shape[1] - test_length

  max_value, max_index = get_value_index(predict[:,:train_length], datas[:,:train_length])

  ans = predict[:,-test_length:]
  ratings = []
  for i in range(predict_num):
    ratings.append(get_ratings(datas[:,-test_length:], max_value[i], max_index[i]))

  print_evalution(ratings, ans)
