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
import layers


class MAFnet:

    # Create model
    def __init__(self, image, sound, target, x_image_shape, x_sound_shape, n_hidden, n_classes):

        self.x_image_shape = x_image_shape
        self.x_sound_shape = x_sound_shape
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.xImage = image
        self.xSound = sound
        self.target = target
        self.isTraining = tf.placeholder(tf.bool, shape=[], name="isTraining")

        # network
        self.logits, self.pred = self.network()

        # loss
        self.loss = self.loss_crossentropy()

        # accuracy
        self.acc = self.accuracy()

    def network(self):
        # FiLM layer
        x_image = tf.reshape(self.xImage, (-1, self.x_image_shape))
        x_sound = tf.reshape(self.xSound, (-1, 12, 8, self.x_sound_shape))

        gamma_from_image, beta_from_image = layers.FILM_generator(x_image, self.n_hidden, name="FILM_for_sound")

        x_sound = layers.FILMed_block(x_sound, self.n_hidden, gamma_from_image, beta_from_image,
                                     self.isTraining, name='FILMed_block_sound')

        x_sound = tf.reduce_mean(x_sound, [1, 2])

        x_image = tf.reshape(x_image, (-1, 10, 1920))
        x_sound = tf.reshape(x_sound, (-1, 10, 512))

        # Dense image
        x_image = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.n_hidden, activation='linear', name='image_fc_1'))(x_image)
        x_image = layers.BN_layer(x_image, self.isTraining, name="image_BN_1")
        x_image = tf.nn.relu(x_image)

        # Dense sound
        x_sound = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.n_hidden, activation='linear', name='sound_fc_1'))(x_sound)
        x_sound = layers.BN_layer(x_sound, self.isTraining, name="sound_BN_1")
        x_sound = tf.nn.relu(x_sound)

        # Modality & temporal Attention Module
        x, self.alpha = layers.modality_temporal_attention(x_image, x_sound, self.n_hidden, 2, self.n_hidden)

        # Dense multimodal
        x = tf.keras.layers.Dense(self.n_classes, activation='linear', use_bias='True', name='multimodal_fc_1')(x)
        x = layers.BN_layer(x, self.isTraining, name="multimodal_BN_1")

        pred = tf.nn.softmax(x)
        return x, pred

    def loss_crossentropy(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
        return loss

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, -1), tf.argmax(self.target, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy