import tensorflow as tf

t = tf.zeros([5, 5, 5, 5])  # creates a tensor of 625 elements

t = tf.reshape(t, [625])  # flatten into a single dim

# Essentially, shape is given by the number of spaces, with each following number being the number of lists and the
# last being the num of elements

print(t)
