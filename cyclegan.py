import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
from model import CycleGAN
import numpy as np
import cv2

################################
#          LOAD DATA           #
################################


# Load H5 files
rgb = h5py.File('/content/gdrive/My Drive/rgbVideos.h5','r')
gray = h5py.File('/content/gdrive/My Drive/grayVideos.h5','r')

# Convert to numpy arrays
rgb = rgb['videos'][()]
gray = gray['videos'][()]

################################
#           TRAINING           #
################################

print('Build graph:')
model = CycleGAN()

variables_to_save = tf.compat.v1.global_variables()
init_op = tf.compat.v1.variables_initializer(variables_to_save)
init_all_op = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    print("Initializing all parameters.")
    sess.run(init_all_op)

    #saver.restore(sess, tf.train.latest_checkpoint('/content/gdrive/My Drive/ckpt'))

    print("Starting training session.")
    model.train(sess, saver, data_a, data_b)

    #model.test(sess,video)