# Copyright 2020 UMONS-Numediart-USHERBROOKE-Necotis.
#
# MAFNet of University of Mons and University of Sherbrooke – Mathilde Brousmiche is free software : you can redistribute it 
# and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the Lesser GNU General Public License for more details. 

# You should have received a copy of the Lesser GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Each use of this software must be attributed to University of MONS – Numédiart Institute and to University of SHERBROOKE - Necotis Lab (Mathilde Brousmiche).

from __future__ import division

from keras.models import Model
from vggish import VGGish
from preprocess_sound import preprocess_sound

import numpy as np
from scipy.io import wavfile
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='audio_feature_extractor')

# Data specifications
parser.add_argument('--data_path', type=str, default="AVE_Dataset",
                    help='data path')
parser.add_argument('--save_path', type=str, default="data",
                    help='save path')
parser.add_argument('--split', type=str, default="test",
                    help='split to extract')
args = parser.parse_args()

def VGGish_extractor(len_video=10, len_window=1):
    sound_model = VGGish(include_top=False, load_weights=True)
    output_layer = sound_model.get_layer(name="conv4/conv4_2").output
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    data_path = args.data_path
    save_path = args.save_path
    split = args.split

    f = open(os.path.join(data_path, split + "Set.txt"), 'r')
    lines = f.read().split('\n')

    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    audio_feature = []
    for n, file in enumerate(lines):
        print( '{}/{}'.format(n, len(lines)-1) )
        if file == '':
            continue  
        if file.split('&')[1] == 'VideoID':
            continue

        video_name = file.split('&')[1]
        # extract sound from video
        os.system("ffmpeg -hide_banner -loglevel panic -i {input} -t {len} {output}.wav".format(
            input=os.path.join(data_path, 'AVE', video_name + '.mp4'),
            output=os.path.join('tmp', video_name), len=len_video))

        sound_path = os.path.join('tmp', video_name + '.wav')
        sr, wav_data = wavfile.read(sound_path)
        input_x = []
        for i in range(len_video):
            tmp = wav_data[sr*i*len_window:sr*(i+1)*len_window]
            if len(tmp.shape) == 2:
                cur_wav = np.zeros((sr, tmp.shape[1]))
                cur_wav[:tmp.shape[0],:] = tmp 
            else:
                cur_wav = np.zeros(sr)
                cur_wav[:tmp.shape[0]] = tmp
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            input_x.append(cur_spectro[0])
        input_x = np.asarray(input_x)
        audio_feature.append(sound_extractor.predict(input_x))
        os.remove(os.path.join('tmp', video_name + '.wav'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    audio_feature = np.asarray(audio_feature)
    output = open(os.path.join(save_path, split + 'Set_audio.p'), 'wb')
    pickle.dump(audio_feature, output, protocol=4)
    return

if __name__ == '__main__':
    VGGish_extractor()

    



