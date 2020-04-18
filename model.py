import os
import random

from tqdm import trange
from imageio import imsave
import tensorflow as tf
import numpy as np

from generator import Generator, Generator1, Generator2
from discriminator import Discriminator, Discriminator1

class CycleGAN(object):
    def __init__(self):
        self._batch_size = 1
        self._image_size = 128
        self._cycle_loss_coeff = 10

        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._image_shape = [self._image_size, self._image_size, 3]

        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)

        image_a = self.image_a = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_a')
        image_b = self.image_b = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_b')
        fake_a = self.fake_a = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._image_shape, name='fake_a')
        fake_b = self.fake_b = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._image_shape, name='fake_b')

        # Data augmentation
        def augment_image(image):
            image = tf.image.resize(image, [self._augment_size, self._augment_size])
            image = tf.image.random_crop(image, [self._batch_size] + self._image_shape)
            image = tf.map_fn(tf.image.random_flip_left_right, image)
            return image

        image_a = tf.cond(self.is_train,
                          lambda: augment_image(image_a),
                          lambda: image_a)
        image_b = tf.cond(self.is_train,
                          lambda: augment_image(image_b),
                          lambda: image_b)

        # Generator
        G_ab = Generator('G_ab', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self._image_size)
        G_ba = Generator('G_ba', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self._image_size)

        # Discriminator
        D_a = Discriminator('D_a', is_train=self.is_train,
                            norm='instance', activation='leaky')
        D_b = Discriminator('D_b', is_train=self.is_train,
                            norm='instance', activation='leaky')

        # Generate images (a->b->a and b->a->b)
        image_ab = self.image_ab = G_ab(image_a)
        image_aba = self.image_aba = G_ba(image_ab)
        image_ba = self.image_ba = G_ba(image_b)
        image_bab = self.image_bab = G_ab(image_ba)

        # Discriminate real/fake images
        D_real_a = D_a(image_a)
        D_fake_a = D_a(image_ba)
        D_real_b = D_b(image_b)
        D_fake_b = D_b(image_ab)
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
        loss_rec_aba = tf.reduce_mean(tf.abs(image_a - image_aba))
        loss_rec_bab = tf.reduce_mean(tf.abs(image_b - image_bab))
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


    def train(self, sess, data_A, data_B):
        print('Start training.')
        print(len(data_A),'images from A')
        print(len(data_B),'images from B')

        data_size = min(len(data_A), len(data_B))
        num_batch = data_size // self._batch_size
        epoch_length = num_batch * self._batch_size

        num_initial_iter = 100
        num_decay_iter = 100
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter


        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:
            #TODO: resume training with global_step
            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)

            if iter == 0:
                random.shuffle(data_A)
                random.shuffle(data_B)

            image_a = np.stack(data_A[iter*self._batch_size:(iter+1)*self._batch_size])
            image_b = np.stack(data_B[iter*self._batch_size:(iter+1)*self._batch_size])
            fake_a, fake_b = sess.run([self.image_ba, self.image_ab],
                                      feed_dict={self.image_a: image_a,
                                                 self.image_b: image_b,
                                                 self.is_train: True})

            fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab,
                       self.loss_G_ba, self.loss_cycle,
                       self.optimizer_D_a, self.optimizer_D_b,
                       self.optimizer_G_ab, self.optimizer_G_ba]

            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.fake_a: fake_a,
                                                   self.fake_b: fake_b})

            if step % 100 == 0:
                 t.set_description(
                    'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                        fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))


    def test(self, sess, data_A, data_B, base_dir):
        step = 0
        for data in data_A:
            step += 1
            fetches = [self.image_ab, self.image_aba]
            image_a = np.expand_dims(data, axis=0)
            image_ab, image_aba = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                    self.is_train: False})
            images = np.concatenate((image_a, image_ab, image_aba), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'a_to_b_{}.jpg'.format(step)), images)

        step = 0
        for data in data_B:
            step += 1
            fetches = [self.image_ba, self.image_bab]
            image_b = np.expand_dims(data, axis=0)
            image_ba, image_bab = sess.run(fetches, feed_dict={self.image_b: image_b,
                                                    self.is_train: False})
            images = np.concatenate((image_b, image_ba, image_bab), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'b_to_a_{}.jpg'.format(step)), images)



class CycleGAN1(object):
    def __init__(self):
        self._batch_size = 1
        self._video_depth = 12
        self._image_size = 256
        self._cycle_loss_coeff = 10

        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15)
        self._color_shape = [self._video_depth, self._image_size, self._image_size, 3]
        self._black_shape = [self._video_depth, self._image_size, self._image_size, 1]

        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        self.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)

        image_a = self.image_a = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._color_shape, name='image_a')
        image_b = self.image_b = \
            tf.compat.v1.placeholder(tf.float32, [self._batch_size] + self._black_shape, name='image_b')
        fake_a = self.fake_a = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._color_shape, name='fake_a')
        fake_b = self.fake_b = \
            tf.compat.v1.placeholder(tf.float32, [None] + self._black_shape, name='fake_b')


        # Generator
        G_ab = Generator1('G_ab', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self._image_size)
        G_ba = Generator2('G_ba', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self._image_size)

        # Discriminator
        D_a = Discriminator1('D_a', is_train=self.is_train,
                            norm='instance', activation='leaky')
        D_b = Discriminator1('D_b', is_train=self.is_train,
                            norm='instance', activation='leaky')

        # Generate images (a->b->a and b->a->b)
        image_ab = self.image_ab = G_ab(image_a)

        image_aba = self.image_aba = G_ba(image_ab)

        image_ba = self.image_ba = G_ba(image_b)

        image_bab = self.image_bab = G_ab(image_ba)

        # Discriminate real/fake images
        D_real_a = D_a(image_a)
        D_fake_a = D_a(image_ba)
        D_real_b = D_b(image_b)
        D_fake_b = D_b(image_ab)
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
        loss_rec_aba = tf.reduce_mean(tf.abs(image_a - image_aba))
        loss_rec_bab = tf.reduce_mean(tf.abs(image_b - image_bab))
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


    def train(self, sess, data_A, data_B):
        print('Start training.')
        print(len(data_A),'images from A')
        print(len(data_B),'images from B')

        data_size = min(len(data_A), len(data_B))
        num_batch = data_size // self._batch_size
        epoch_length = num_batch * self._batch_size

        num_initial_iter = 1
        num_decay_iter = 1
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter


        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:
            #TODO: resume training with global_step

            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)
            '''
            if iter == 0:
                random.shuffle(data_A)
                random.shuffle(data_B)

            image_a = np.stack(data_A[iter*self._batch_size:(iter+1)*self._batch_size])
            image_b = np.stack(data_B[iter*self._batch_size:(iter+1)*self._batch_size])
            '''
            image_a = data_A
            image_b = data_B
            
            fake_a, fake_b = sess.run([self.image_ba, self.image_ab],
                                      feed_dict={self.image_a: image_a,
                                                 self.image_b: image_b,
                                                 self.is_train: True})

            print("here1")
            fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab,
                       self.loss_G_ba, self.loss_cycle,
                       self.optimizer_D_a, self.optimizer_D_b,
                       self.optimizer_G_ab, self.optimizer_G_ba]

            print("here2")
            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.fake_a: fake_a,
                                                   self.fake_b: fake_b})

            print("here3")
            if step % 100 == 0:
                 t.set_description(
                    'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                        fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))


    def test(self, sess, data_A, data_B, base_dir):
        step = 0
        for data in data_A:
            step += 1
            fetches = [self.image_ab, self.image_aba]
            image_a = np.expand_dims(data, axis=0)
            image_ab, image_aba = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                    self.is_train: False})
            images = np.concatenate((image_a, image_ab, image_aba), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'a_to_b_{}.jpg'.format(step)), images)

        step = 0
        for data in data_B:
            step += 1
            fetches = [self.image_ba, self.image_bab]
            image_b = np.expand_dims(data, axis=0)
            image_ba, image_bab = sess.run(fetches, feed_dict={self.image_b: image_b,
                                                    self.is_train: False})
            images = np.concatenate((image_b, image_ba, image_bab), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'b_to_a_{}.jpg'.format(step)), images)