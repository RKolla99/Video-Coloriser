import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
from model import CycleGAN, CycleGAN1
from data_loader import get_data
import numpy as np
import cv2
import ops

#print('Read data:')
#train_A, train_B, test_A, test_B = get_data('apple2orange', 128)

cap = cv2.VideoCapture('movie.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fc=0
ret=True
buf = np.empty((12, 256, 256, 3), np.dtype('uint8'))
bnw = np.empty((12, 256, 256), np.dtype('uint8'))

while (fc < frameCount  and ret):
    ret, frame = cap.read()
    buf[fc] = cv2.resize(frame,(256,256))
    bnw[fc] = cv2.cvtColor(buf[fc], cv2.COLOR_BGR2GRAY)
    fc += 1
    if (fc==12):
    	break
cap.release()

'''
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (256,256))
for i in range(len(buf)):
	out.write(buf[i])
out.release()
'''
bnw = np.expand_dims(bnw,axis=3)
bnw = np.expand_dims(bnw,axis=0)
buf = np.expand_dims(buf,axis=0)
print(np.shape(buf))

'''
#colour to black and white
G_ab = Generator1('G_ab', is_train=True, 
                 norm='instance', activation='relu', image_size=256)
G_ba = Generator2('G_ba', is_train=True, 
                 norm='instance', activation='relu', image_size=256)

fake_buf = G_ab(tf.convert_to_tensor(buf,dtype=tf.float32))
cycle_buf = G_ba(tf.convert_to_tensor(fake_buf,dtype=tf.float32))

D_a = Discriminator('D_a', is_train=True,    #input 1, output 3 channel
                    norm='instance', activation='leaky')
D_b = Discriminator('D_b', is_train=True, #input 3, output 1 channel
                    norm='instance', activation='leaky')

D_real_a = D_a(tf.convert_to_tensor(fake_buf,dtype=tf.float32))
D_real_b = D_b(tf.convert_to_tensor(cycle_buf,dtype=tf.float32))
'''

print('Build graph:')
model = CycleGAN1()

variables_to_save = tf.compat.v1.global_variables()
init_op = tf.compat.v1.variables_initializer(variables_to_save)
init_all_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    print("Initializing all parameters.")
    sess.run(init_all_op)

    print("Starting training session.")
    model.train(sess, buf, bnw)