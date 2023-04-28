# 1) Linear regression and logistic regression
# 2) Convolutional Neural Networks (CNNs)
# 3) Recurrent Neural Networks (RNNs): Sequences, Long-Short Term Memory, language modelling
# 4) Unsupervised learning and Restricted Boltzmann Machines (RBMs): a movie recommendation
# 5) Unsupervised learning and Autoencoders: detecting patterns

import tensorflow as tf


###
### New method
###
assert tf.executing_eagerly() == True

a = tf.constant([1, 2, 3, 4])
b = tf.constant(np.array([5, 6, 7, 8]))
c = tf.Variable([8, 9, 10, 11])
d = tf.convert_to_tensor([123, 456,])
back_to_numpy = a.numpy()

operations = tf.matmul, tf.add, tf.subtract, tf.nn.sigmoid, ...
dot_product = tf.tensordot(a, b, axes=1)
addition_result = c + 1

@tf.function
def add(first, second):
    c = tf.add(first, second)
    print(c)
    return c
result_of_function = add(a, b)

###
### Old method
###
from tensorflow.python.framework.ops import disable_eager_execution; disable_eager_execution()
assert tf.executing_eagerly() == False

a = tf.constant([1, 2, 3, 4])
b = tf.constant(np.array([5, 6, 7, 8]))
dot_product = tf.tensordot(a, b, axes=1)

with tf.compat.v1.Session() as sess: # sess = tf.compat.v1.Session()
    output = sess.run(dot_product) # evaluate result of variable
    # sess.close()