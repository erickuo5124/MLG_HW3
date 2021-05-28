# Item-based Collaborative Filtering with consine similarity
import numpy as np
import pandas as pd
from utils.evaluation import print_evalution
from utils.CF import get_ratings, get_value_index_cos

# parameter
path = './datas/processed/movie/user_movie.pkl'
predict_num = 100 # number of predict users
test_size = 0.2

if __name__ == '__main__':
  df = pd.read_pickle(path)

  datas = df.to_numpy().astype('int')
  datas = np.swapaxes(datas, 0, 1)

  user_num = datas.shape[0]
  rnd = np.random.randint(user_num, size=predict_num)
  predict = datas[rnd]
  datas = np.delete(datas, rnd, 0)

  test_length = int(datas.shape[1] * test_size)
  train_length = datas.shape[1] - test_length

  max_value, max_index = get_value_index_cos(predict[:,:train_length], datas[:,:train_length])

  ans = predict[:,-test_length:]
  ratings = []
  for i in range(predict_num):
    ratings.append(get_ratings(datas[:,-test_length:], max_value[i], max_index[i]))

  print_evalution(ratings, ans)