import tensorflow as tf
import numpy as np
w = 5
LEN = 256
self.window_mask = np.zeros((LEN,LEN))
alp = 0.1

upper = LEN - (2 * w + 1)
for i in range(16):
    self.window_mask[LEN-i-1,:] = (1) 
    self.window_mask[:,LEN-i-1] = (1) 

for i in range(LEN):

    low_bound = max(i-w, 0)
    upper_bound = min(i+w+1, LEN)
    alp_low = max(0-(i-w), 0)
    alp_upper = min(2*w+1, LEN+w-i)
    self.window_mask[i,low_bound:upper_bound] = (1) 

    rand_val = np.random.randint(0, upper, size=2)
    for r_val in rand_val:
        if r_val >= low_bound and r_val <= upper_bound:
            r_val = r_val + 2 * w + 1
        self.window_mask[i,r_val] = 1

import pdb
pdb.set_trace()
self.window_mask = tf.cast(tf.convert_to_tensor(self.window_mask), tf.float32)

    # print(i)
# print(self.window_mask)

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer()) 
# #   import pdb
# #   pdb.set_trace()
#   print(sess.run(self.window_mask[1]))
#   print(sess.run(self.window_mask))