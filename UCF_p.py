# User-based Collaborative Filtering with pearson correlation similarity
import numpy as np
import pandas as pd
from utils.evaluation import print_evalution
from utils.CF import get_min_index, get_ratings
from utils.similarity import pearson_sim

# parameter
path = './datas/processed/movie/user_movie.pkl'
predict_num = 5 # number of predict users
test_size = 0.2
k = 10 # k most similar users

def get_value_index(predict, datas):
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
