# Copyright 2020 UMONS-Numediart-USHERBROOKE-Necotis.
#
# MAFNet of University of Mons and University of Sherbrooke – Mathilde Brousmiche is free software : you can redistribute it 
# and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the Lesser GNU General Public License for more details. 

# You should have received a copy of the Lesser GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Each use of this software must be attributed to University of MONS – Numédiart Institute and to University of SHERBROOKE - Necotis Lab (Mathilde Brousmiche).

from keras.applications.densenet import DenseNet201, preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical

import numpy as np
import os
import argparse
import pickle
import shutil

parser = argparse.ArgumentParser(description='visual_feature_extractor')

# Data specifications
parser.add_argument('--data_path', type=str, default="AVE_Dataset",
                    help='data path')
parser.add_argument('--save_path', type=str, default="data",
                    help='save path')
parser.add_argument('--split', type=str, default="test",
                    help='split to extract')
args = parser.parse_args()

classes_AVE = {
            'Accordion' : 0,
            'Acoustic guitar' : 1,
            'Baby cry, infant cry' : 2,
            'Banjo' : 3,
            'Bark' : 4,
            'Bus' : 5,
            'Cat' : 6,
            'Chainsaw' : 7,
            'Church bell' : 8,
            'Clock' : 9,
            'Female speech, woman speaking' : 10,
            'Fixed-wing aircraft, airplane' : 11,
            'Flute' : 12,
            'Frying (food)' : 13,
            'Goat' : 14,
            'Helicopter' : 15,
            'Horse' : 16,
            'Male speech, man speaking' : 17,
            'Mandolin' : 18,
            'Motorcycle' : 19,
            'Race car, auto racing' : 20,
            'Rodents, rats, mice' : 21,
            'Shofar' : 22,
            'Toilet flush' : 23,
            'Train horn' : 24,
            'Truck' : 25,
            'Ukulele' : 26,
            'Violin, fiddle' : 27
                }


def DenseNet_extractor(fps=16, len_video=10, len_window=1):

    model = DenseNet201(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3))

    data_path = args.data_path
    save_path = args.save_path
    split = args.split

    f = open(os.path.join(data_path, split + "Set.txt"), 'r')
    lines = f.read().split('\n')

    video_feature = []
    target = []
    for n, file in enumerate(lines):
        print('{}/{}'.format(n,len(lines)-1))
        if file == '':
            continue

        video_name = file.split('&')[1]
        if not os.path.exists(os.path.join('tmp', video_name)):
            os.makedirs(os.path.join('tmp', video_name))

        # extract frame from video
        os.system("ffmpeg -hide_banner -loglevel panic -i {input} -vf  fps={fps} -t {len} {output}%04d.jpg".format(
                                                input=os.path.join(data_path, 'AVE', video_name + '.mp4'),
                                                output=os.path.join('tmp', video_name, 'img'),
                                                fps=fps, len=len_video))

        if not os.path.exists(os.path.join('tmp', video_name, 'img0160.jpg')):
            shutil.copyfile(os.path.join('tmp', video_name, 'img0159.jpg'), os.path.join('tmp', video_name, 'img0160.jpg'))


        # extract feature
        input_x = []
        for j in range(len_video*fps):
            img_path = os.path.join('tmp', video_name, 'img' + '{:04d}.jpg'.format(j+1))
            x = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(x)
            x = preprocess_input(x)
            input_x.append(x)

        input_x = np.asarray(input_x)
        features = model.predict(input_x)
        features = np.mean(features, axis = (1,2))
        tmp = []
        for j in range(len_video):
            tmp.append(np.mean(features[fps*j:fps*(j+1),:], axis=0))
        video_feature.append(tmp)

        shutil.rmtree(os.path.join('tmp', video_name))

        target.append(to_categorical(classes_AVE[file.split('&')[0]], num_classes=28))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video_feature = np.asarray(video_feature)
    output = open(os.path.join(save_path, split + 'Set_visual.p'), 'wb')
    pickle.dump(video_feature, output)

    target = np.asarray(target)
    output = open(os.path.join(save_path, split + 'Set_target.p'), 'wb')
    pickle.dump(target, output)
    return


if __name__ == '__main__':
  DenseNet_extractor()
