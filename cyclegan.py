import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
from model import CycleGAN
from data_loader import get_data

print('Read data:')
train_A, train_B, test_A, test_B = get_data('apple2orange', 128)

print('Build graph:')
model = CycleGAN()

variables_to_save = tf.compat.v1.global_variables()
init_op = tf.compat.v1.variables_initializer(variables_to_save)
init_all_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    print("Initializing all parameters.")
    sess.run(init_all_op)

    print("Starting training session.")
    model.train(sess, train_A, train_B)
