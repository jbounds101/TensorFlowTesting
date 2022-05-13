import tensorflow as tf

# Tensor: a vector generalized for potentially higher dimensions

""" How to create tensors (these are scalars, think of one dimension vectors), these are rank 0
string = tf.Variable('string here!', tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(1.1231, tf.float64)
"""

""" Rank 1 and 2 tensors
rank1 = tf.Variable(['test', '0'], tf.string)
    Can only have one pivot, it is a single vector

rank2 = tf.Variable([['test', 'ok'], ['test', 'yes']], tf.string)
    Two pivots, two vectors, rank = 2!

tf.rank(rank2)  # can be used to find the rank (numpy gives rank)

rank2.shape  # gives the m x n shape of the tensor [2, 2]

# shape is of the form: (number of exterior lists, number of lists inside that list, number of elements in those lists)

"""

tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to the new shape
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension of that place,
# in this case [3, 2]
print(tensor1)
print(tensor2)
print(tensor3)

""" Tensor types
Variable
Constant
Placeholder
SparseTensor

All these tensors are immutable except for Variable
"""


""" Evaluate a tensor
with tf.Session() as sess:
    tensor1.eval()
"""