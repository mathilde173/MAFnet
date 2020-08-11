# Copyright 2020 UMONS-Numediart-USHERBROOKE-Necotis.
#
# MAFNet of University of Mons and University of Sherbrooke – Mathilde Brousmiche is free software : you can redistribute it 
# and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the Lesser GNU General Public License for more details. 

# You should have received a copy of the Lesser GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Each use of this software must be attributed to University of MONS – Numédiart Institute and to University of SHERBROOKE - Necotis Lab (Mathilde Brousmiche).

import tensorflow as tf
import os
import sklearn
import pickle
import numpy as np
import argparse
import model

parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--data_path', type=str, default="data",
                    help='data path')

parser.add_argument('--n_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=32,
                    help='number of batch size')
parser.add_argument('--learning_rate', type=float, default=1E-4,
                    help='number of batch size')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for early stopping')

parser.add_argument('--x_image_shape', type=int, default=1920,
                    help='image feature size')
parser.add_argument('--x_sound_shape', type=int, default=512,
                    help='sound feature size')
parser.add_argument('--n_classes', type=int, default=28,
                    help='number of classes')
parser.add_argument('--n_hidden', type=int, default=512,
                    help='number of hidden neurons')

parser.add_argument('--prob_img', type=float, default=0.,
                    help='rate for updating weights from image pathway')
parser.add_argument('--prob_all', type=float, default=.9,
                    help='rate for updating weights from both pathways')

parser.add_argument('--train', action='store_true', default=False,
                    help='train a new model')
args = parser.parse_args()


def load_data(path, set, shuffle=True):
    print('Load data ' + set + 'set')
    pkl_file = open(os.path.join(path, set+'Set_visual.p'), 'rb')
    visual = pickle.load(pkl_file)
    visual = np.asarray(visual)

    pkl_file = open(os.path.join(path, set + 'Set_audio.p'), 'rb')
    audio = pickle.load(pkl_file)
    audio = np.asarray(audio)

    pkl_file = open(os.path.join(path, set + 'Set_target.p'), 'rb')
    target = pickle.load(pkl_file)
    target = np.asarray(target)

    if shuffle:
        visual, audio, target = sklearn.utils.shuffle(visual, audio, target)

    return visual, audio, target


def train_model(train_data, val_data):
    print('Training ...')

    # Data Generator
    x_image = tf.placeholder(tf.float32, shape=(None, 10, args.x_image_shape), name='x_image')
    x_sound = tf.placeholder(tf.float32, shape=(None, 10, 12, 8, args.x_sound_shape), name='x_sound')
    y = tf.placeholder(tf.float32, shape=(None, args.n_classes), name='target')

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)

    # setup network
    MAFnet = model.MAFnet(x_image, x_sound, y, args.x_image_shape, args.x_sound_shape, args.n_hidden, args.n_classes)

    var_list_image = []
    var_list_sound = []
    for var in tf.trainable_variables():
        if 'image' in var.name:
            var_list_image.append(var)
        if 'sound' in var.name:
            var_list_sound.append(var)
        if (not 'image' in var.name) and not ('sound' in var.name):
            var_list_image.append(var)
            var_list_sound.append(var)

    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss)
    train_step_image = tf.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss, var_list=var_list_image)
    train_step_sound = tf.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss, var_list=var_list_sound)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Train initialization
    min_delta = 0.0001
    patience_cnt = 0
    best_acc_val = 0
    best_epoch = 0



    N = len(train_data[0])
    for n in range(args.n_epoch):

        total_loss = 0
        train_data = sklearn.utils.shuffle(train_data[0], train_data[1], train_data[2])
        for i in range(int(N / args.batch_size)):
            random_choice = np.random.choice(['image', 'sound', 'all'], size=int(N / args.batch_size),
                                             p=[args.prob_img, 1 - args.prob_img - args.prob_all, args.prob_all])

            feed_dict = {x_image: train_data[0][i*args.batch_size:(i+1)*args.batch_size],
                         x_sound: train_data[1][i*args.batch_size:(i+1)*args.batch_size],
                         y: train_data[2][i*args.batch_size:(i+1)*args.batch_size],
                         MAFnet.isTraining: True}

            if random_choice[i] == 'image':
                _, l = sess.run([train_step_image, MAFnet.loss], feed_dict=feed_dict)
            elif random_choice[i] == 'sound':
                _, l = sess.run([train_step_sound, MAFnet.loss], feed_dict=feed_dict)
            elif random_choice[i] == 'all':
                _, l = sess.run([train_step, MAFnet.loss], feed_dict=feed_dict)

            total_loss += l

        acc_val = sess.run(MAFnet.acc, feed_dict={ x_image: val_data[0],
                                               x_sound: val_data[1],
                                               y: val_data[2],
                                               MAFnet.isTraining: False})

        # Early stopping
        if n > 0 and (acc_val - best_acc_val) > min_delta:
            patience_cnt = 0
            best_acc_val = acc_val
            best_epoch = n
            saver.save(sess, os.path.join('model', 'MAFnet'))
        else:
            patience_cnt += 1

        print('>> Epoch [{}/{}] : Accuracy val {:.2f} Best epoch : {}'.format(n + 1, args.n_epoch,
                                                                              acc_val * 100, best_epoch+1))
        if patience_cnt > args.patience:
            print("Early stopping...")
            print("Best epoch : " + str(best_epoch+1))
            break
    return

def test_model(test_data):
    print('Testing ...')
    # Data Generator
    x_image = tf.placeholder(tf.float32, shape=(None, 10, args.x_image_shape), name='x_image')
    x_sound = tf.placeholder(tf.float32, shape=(None, 10, 12, 8, args.x_sound_shape), name='x_sound')
    y = tf.placeholder(tf.float32, shape=(None, args.n_classes), name='target')

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)

    # setup network
    MAFnet = model.MAFnet(x_image, x_sound, y, args.x_image_shape, args.x_sound_shape, args.n_hidden, args.n_classes)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('model', 'MAFnet'))
    accuracy = sess.run(MAFnet.acc, feed_dict={x_image: test_data[0],
                                               x_sound: test_data[1],
                                               y: test_data[2],
                                               MAFnet.isTraining: False})
    print('Accuracy : {}'.format(accuracy))
    return


if __name__=='__main__':
    if args.train == True:
        train_data = load_data(args.data_path, set='train')
        val_data = load_data(args.data_path, set='val')
        train_model(train_data, val_data)
    else:
        test_data = load_data(args.data_path, set='test', shuffle=False)
        test_model(test_data)
