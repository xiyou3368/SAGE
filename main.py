import numpy as np
from one_fold import fold_cv
import tensorflow as tf



temp = []
for i in range(10):
  tf.reset_default_graph()
  seed = 4
  np.random.seed(seed)
  tf.set_random_seed(seed)
  ttemp = fold_cv(cv_index = i)
  temp.append(ttemp)
  print i,ttemp
print("the average is ","{:.5f}".format(sum(temp)/10)) 
