import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

def cos_sim(u, v):
  a = dot(u, v)
  b = norm(u)
  c = norm(v)
  return a / (b * c) if b and c else 1

def pearson_sim(u, v):
  avg_ru = np.average(u)
  avg_rv = np.average(v)
  a = np.sum([(ru-avg_ru)*(rv-avg_rv) for ru, rv in zip(u, v)])
  b = math.sqrt(np.sum([pow(ru-avg_ru,2) for ru in u]))
  c = math.sqrt(np.sum([pow(rv-avg_rv,2) for rv in v]))
  return a / (b * c) if b and c else 1