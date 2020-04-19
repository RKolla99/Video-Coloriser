import os
import random

from tqdm import trange
from imageio import imsave
import tensorflow as tf
import numpy as np

from generator import Generator
from discriminator import Discriminator
import cv2

class CycleGAN(object):
    def __init__(self):
        self._batch_size = 1
        self._video_depth = 12
        self._image_size = 256
        self._cycle_loss_coeff = 10

        self._color_shape = [self._video_depth, self._image_size, self._image_size, 3]
        self._black_shape = [self._video_depth, self._image_size, self._image_size, 1]

        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)

        video_a = self.video_a = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._color_shape, name='video_a')
        video_b = self.video_b = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._black_shape, name='video_b')
        fake_a = self.fake_a = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._color_shape, name='fake_a')
        fake_b = self.fake_b = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._black_shape, name='fake_b')


        # Generator
        G_ab = self.G_ab = Generator('G_ab', is_train=self.is_train,
                         norm='instance', activation='relu', 
                         image_size=self._image_size, output_channels=1)
        G_ba = self.G_ba = Generator('G_ba', is_train=self.is_train,
                         norm='instance', activation='relu', 
                         image_size=self._image_size, output_channels=3)

        # Discriminator
        D_a = self.D_a = Discriminator('D_a', is_train=self.is_train,
                            norm='instance', activation='leaky')
        D_b = self.D_b = Discriminator('D_b', is_train=self.is_train,
                            norm='instance', activation='leaky')

        # Generate videos (a->b->a and b->a->b)
        video_ab = self.video_ab = G_ab(video_a)

        video_aba = self.video_aba = G_ba(video_ab)

        video_ba = self.video_ba = G_ba(video_b)

        video_bab = self.video_bab = G_ab(video_ba)

        # Discriminate real/fake videos
        D_real_a = D_a(video_a)
        D_fake_a = D_a(video_ba)
        D_real_b = D_b(video_b)
        D_fake_b = D_b(video_ab)
        D_fake_a = D_a(fake_a)
        D_fake_b = D_b(fake_b)

        # Least squre loss for GAN discriminator
        loss_D_a = (tf.reduce_mean(tf.math.squared_difference(D_real_a, 0.9)) +
            tf.reduce_mean(tf.square(D_fake_a))) * 0.5
        loss_D_b = (tf.reduce_mean(tf.math.squared_difference(D_real_b, 0.9)) +
            tf.reduce_mean(tf.square(D_fake_b))) * 0.5

        # Least squre loss for GAN generator
        loss_G_ab = tf.reduce_mean(tf.math.squared_difference(D_fake_b, 0.9))
        loss_G_ba = tf.reduce_mean(tf.math.squared_difference(D_fake_a, 0.9))

        # L1 norm for reconstruction error
        loss_rec_aba = tf.reduce_mean(tf.abs(video_a - video_aba))
        loss_rec_bab = tf.reduce_mean(tf.abs(video_b - video_bab))
        loss_cycle = self._cycle_loss_coeff * (loss_rec_aba + loss_rec_bab)

        loss_G_ab_final = loss_G_ab + loss_cycle
        loss_G_ba_final = loss_G_ba + loss_cycle

        # Optimizer
        self.optimizer_D_a = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_D_a, var_list=D_a.var_list, global_step=self.global_step)
        self.optimizer_D_b = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_D_b, var_list=D_b.var_list)
        self.optimizer_G_ab = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_G_ab_final, var_list=G_ab.var_list)
        self.optimizer_G_ba = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_G_ba_final, var_list=G_ba.var_list)

        # Summaries
        self.loss_D_a = loss_D_a
        self.loss_D_b = loss_D_b
        self.loss_G_ab = loss_G_ab
        self.loss_G_ba = loss_G_ba
        self.loss_cycle = loss_cycle


    def train(self, sess, saver, data_A, data_B):
        print('Start training.')
        print(len(data_A),'videos from A')
        print(len(data_B),'videos from B')

        data_size = min(len(data_A), len(data_B))
        num_batch = data_size // self._batch_size
        epoch_length = num_batch * self._batch_size

        num_initial_iter = 5
        num_decay_iter = 5
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter

        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:

            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)

            if iter == 0:
                random.shuffle(data_A)
                random.shuffle(data_B)

            video_a = np.expand_dims(data_A[iter],axis=0)
            video_b = np.expand_dims(data_B[iter],axis=0)
            video_b = np.expand_dims(video_b,axis=4)
            
            fake_a, fake_b = sess.run([self.video_ba, self.video_ab],
                                      feed_dict={self.video_a: video_a,
                                                 self.video_b: video_b,
                                                 self.is_train: True})

            fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab,
                       self.loss_G_ba, self.loss_cycle,
                       self.optimizer_D_a, self.optimizer_D_b,
                       self.optimizer_G_ab, self.optimizer_G_ba]

            fetched = sess.run(fetches, feed_dict={self.video_a: video_a,
                                                   self.video_b: video_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.fake_a: fake_a,
                                                   self.fake_b: fake_b})

            t.set_description(
                'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                    fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))

           if (step%60 == 0):
                saver.save(sess, '/content/gdrive/My Drive/ckpt/3dcyclegan')

    def test(self, sess, vid):

        fetches = [self.video_ba]
        video_ba = sess.run(fetches, feed_dict={self.video_b: vid,
                                            self.is_train:False})

        video_ba = np.squeeze(video_ba)

        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (256,256))
        for i in range(len(video_ba)):
            out.write(video_ba[i])
        out.release()