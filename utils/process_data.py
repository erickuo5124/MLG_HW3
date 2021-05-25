import numpy as np
import pandas as pd
from os import listdir

def process(path, rows, cols):
  data = pd.read_csv(path, sep='\t', header=None)
  df = pd.DataFrame(np.zeros((rows, cols)))
  interaction_count = np.zeros(rows)
  drop_index = []
  num_data = data.shape[0]

  for index, row in data.iterrows():
    if len(row) >= 3: # data with rating
      df[row[1]-1][row[0]-1] = row[2]
    else: # one hot data
      df[row[1]-1][row[0]-1] = 1
    interaction_count[row[0]-1] += 1
    if index%10000 == 0:
      print(f'{(index/(num_data/10)*10)}%')

  print('filtering...')
  for index, count in enumerate(interaction_count):
    if count < 3:
      drop_index.append(index)
  
  # print(f'Drop index: {drop_index}')
  df.drop(drop_index, inplace=True)
  return df


if __name__ == '__main__':
  print('data processing...')

  # raw_path = '../datas/raw'
  # processed_path = '../datas/processed'
  # datasets = listdir(raw_path)
  # for dataset in datasets:
  #   files = listdir(f'{raw_path}/{dataset}')
  #   for file in files:
  #     print(f'{raw_path}/{dataset}/{file}')
  #     print(f'{processed_path}/{dataset}/{file[:-4]}.pkl')
      
  process('../datas/raw/book/user_book.dat', 13204, 22347).to_pickle('../datas/processed/book/user_book.pkl')
  print('finish!')