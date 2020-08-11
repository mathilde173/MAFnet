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
import numpy as np

# Batch Normalization
def batch_norm(input, isTraining, reuse=None, name="BN"):
  with tf.variable_scope(name, reuse=reuse):
    decay = 0.999
    shape = input.shape
    pop_mean = tf.get_variable('pop_mean', shape = shape[-1], initializer=tf.constant_initializer(0.0),trainable = False)
    pop_var = tf.get_variable('pop_var', shape=shape[-1], initializer=tf.constant_initializer(1.0), trainable = False)
    with tf.name_scope(name):
      scale = tf.get_variable('scale', shape=shape[-1], initializer=tf.constant_initializer(1.0))
      beta = tf.get_variable('beta', shape=shape[-1], initializer=tf.constant_initializer(0.0))

      if len(shape) == 4:
          newShape = [-1, 1, 1, shape[-1].value]
      elif len(shape) == 3:
          newShape = [-1, 1, shape[-1].value]
      else:
          newShape = [-1, shape[-1].value]

      if isTraining:
        batch_mean, batch_var = tf.nn.moments(input,list(np.arange(len(shape)-1)))
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
          BN = tf.nn.batch_normalization(input, batch_mean, batch_var, tf.reshape(beta, newShape), tf.reshape(scale, newShape), 1E-3)
      else:
        BN = tf.nn.batch_normalization(input, pop_mean, pop_var, tf.reshape(beta, newShape), tf.reshape(scale, newShape), 1E-3)
      return BN


def BN_layer(input, isTraining, name="BN"):
    BN = tf.cond(isTraining,
                  lambda: batch_norm(input, True, None, name),
                  lambda: batch_norm(input, False, True, name))
    return BN


# FiLM layer
def FILM_generator(x, n_hidden, name="FILM"):
  FILM = tf.keras.layers.Dense(2 * n_hidden, activation='linear', name=name + "_fc")(x)
  gamma, beta = tf.split(FILM, [n_hidden, n_hidden], 1)
  return gamma + 1, beta


def FILM_application(x, gamma, beta):
  shape = tf.shape(x)

  gamma = tf.expand_dims(gamma, 1)
  gamma = tf.expand_dims(gamma, 2)
  gamma = tf.tile(gamma, [1, shape[1], shape[2], 1])

  beta = tf.expand_dims(beta, 1)
  beta = tf.expand_dims(beta, 2)
  beta = tf.tile(beta, [1, shape[1], shape[2], 1])

  return gamma * x + beta


def FILMed_block(x, n_filter, gamma, beta, isTraining, name='FILMed_block'):
  x = tf.keras.layers.Conv2D(n_filter, (1,1), strides=(1, 1), padding="SAME", name=name + "_conv1")(x)
  x = tf.nn.relu(x)

  res = x

  x = tf.keras.layers.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="SAME", name=name + "_conv2")(x)
  x = BN_layer(x, isTraining, name=name + '_BN')
  x = FILM_application(x, gamma, beta)
  x = tf.nn.relu(x)

  x = x + res
  return x


# Modality & Temporal Attention Module
def modality_temporal_attention(image_feature, sound_feature, feature_shape, n_modality, n_hidden):
  U = tf.keras.layers.Dense(n_hidden, activation='relu')
  W = tf.keras.layers.Dense(1, activation='linear')

  V = tf.concat([tf.expand_dims(image_feature, axis=2), tf.expand_dims(sound_feature, axis=2)], axis=2)

  v_t = tf.concat([image_feature, sound_feature], axis=-1)
  v_t = tf.reshape(v_t, (-1, feature_shape))


  v_t = U(v_t)
  x_t = W(v_t)


  x_t = tf.reshape(x_t, (-1, n_modality * 10))

  w_t = tf.nn.softmax(x_t, axis=-1)  ## attention map

  w_t = tf.reshape(w_t, (-1, 10, n_modality, 1))

  attention_vector = w_t * V

  out = tf.concat([attention_vector[:,:,0,:], attention_vector[:,:,1,:]], axis=-1)
  out = tf.reduce_sum(out, axis=1)

  return out, w_t