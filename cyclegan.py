import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
from model import CycleGAN
import numpy as np
import cv2
import ops


################################
#       INPUT PIPELINE         #
################################


data_a = []
data_b = []
for i in range(3):
	cap = cv2.VideoCapture('movie.mp4')
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fc=0
	ret=True
	buf = np.empty((12, 256, 256, 3), np.dtype('uint8'))
	bnw = np.empty((12, 256, 256), np.dtype('uint8'))
	while (fc < frameCount  and ret):
	    ret, frame = cap.read()
	    buf[fc%12] = cv2.resize(frame,(256,256))
	    bnw[fc%12] = cv2.cvtColor(buf[fc%12], cv2.COLOR_BGR2GRAY)
	    if (fc%12==0):
	    	data_a.append(buf)
	    	data_b.append(bnw)
	    	buf = np.empty((12, 256, 256, 3), np.dtype('uint8'))
	    	bnw = np.empty((12, 256, 256), np.dtype('uint8'))
	    fc += 1
	cap.release()

print(np.shape(data_a),np.shape(data_b))


'''
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (256,256))
for i in range(len(buf)):
	out.write(buf[i])
out.release()
'''

'''
cap = cv2.VideoCapture('movie.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fc=0
ret=True
buf = np.empty((12, 256, 256, 3), np.dtype('uint8'))
bnw = np.empty((12, 256, 256), np.dtype('uint8'))
while (fc < frameCount  and ret):
    ret, frame = cap.read()
    buf[fc%12] = cv2.resize(frame,(256,256))
    bnw[fc%12] = cv2.cvtColor(buf[fc%12], cv2.COLOR_BGR2GRAY)
    if (fc%12==0):
    	break
    fc += 1
cap.release()

bnw = np.expand_dims(bnw,axis=3)
bnw = np.expand_dims(bnw,axis=0)
buf = np.expand_dims(buf,axis=0)

print(np.shape(buf),np.shape(bnw))
'''

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

    print("Starting training session.")
    model.train(sess, saver, data_a, data_b)
    #model.test(sess,bnw)