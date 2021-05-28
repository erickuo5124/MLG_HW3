# Matrix Factorization
import numpy as np
import pandas as pd
from utils.evaluation import print_evalution
import torch
from torch.nn.functional import relu

# parameter
path = './datas/processed/movie/user_movie.pkl'
predict_num = 100 # number of predict users
test_size = 0.2
k = 10 # k-dim latent factor
LR = 0.01
EPOCH_NUM = 10

class user_Net(torch.nn.Module):
  def __init__(self, n_feature):
    super(user_Net, self).__init__()
    self.user_para = torch.nn.Linear(n_feature, n_feature)
  
  def forward(self, user_feat, item_feat):
    user_feat = self.user_para(user_feat)
    result = torch.matmul(user_feat, torch.swapaxes(item_feat, 0, 1))

    return user_feat, result

class item_Net(torch.nn.Module):
  def __init__(self, n_feature):
    super(item_Net, self).__init__()
    self.item_para = torch.nn.Linear(n_feature, n_feature)
  
  def forward(self, user_feat, item_feat):
    item_feat = self.item_para(item_feat)
    result = torch.matmul(user_feat, torch.swapaxes(item_feat, 0, 1))

    return item_feat, result

def MF(m):
  user_feat = torch.rand(m.shape[0], k)
  item_feat = torch.rand(m.shape[1], k)
  m = torch.from_numpy(m).float()

  user_model = user_Net(k)
  item_model = item_Net(k)
  criterion = torch.nn.MSELoss()
  user_optim = torch.optim.Adam(user_model.parameters(), lr=LR)
  item_optim = torch.optim.Adam(item_model.parameters(), lr=LR)
  for epoch in range(EPOCH_NUM):
    user_optim.zero_grad()
    user_feat, result = user_model(user_feat, item_feat)
    loss = criterion(result, m)
    loss.backward(retain_graph=True)
    user_optim.step()

    item_optim.zero_grad()
    item_feat, result = item_model(user_feat, item_feat)
    loss = criterion(result, m)
    loss.backward(retain_graph=True)
    item_optim.step()

    # avoid inplace operation
    user_feat = user_feat.detach()
    item_feat = item_feat.detach()
    # print(f'Epoch: {epoch}, Loss: {loss}')
  return user_feat.numpy(), torch.swapaxes(item_feat, 0, 1).numpy()

if __name__ == '__main__':
  df = pd.read_pickle(path)

  datas = df.to_numpy().astype('int')

  user_num = datas.shape[0]
  test_length = int(datas.shape[1] * test_size)
  train_length = datas.shape[1] - test_length
  rnd_user = np.random.randint(user_num, size=predict_num)

  user_feat, _ = MF(datas[:,:train_length])
  _, item_feat = MF(np.delete(datas, rnd_user, 0))

  predict = np.matmul(user_feat, item_feat)

  print_evalution(
    predict[rnd_user, -test_length:],
    datas[rnd_user, -test_length:]
  )

  # print(user_feat)
  # print()
  # print(item_feat)

  # print(
  #   predict[rnd_user, -test_length:],
  #   datas[rnd_user, -test_length:]
  # )