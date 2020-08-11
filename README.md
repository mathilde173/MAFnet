# MAFnet

Multi-level Attention Fusion Network (MAFnet) is a multimodal network that can fuse dynamically visual and audio information for audio-visual event recognition.

We release the testing code along trained models.

## Authors

- Mathilde Brousmiche (<mathilde.brousmiche@umons.ac.be>)
- Stéphane Dupont (<stephane.dupont@umons.ac.be>)
- Jean Rouat (<jean.rouat@usherbrooke.ca>)

## MAFnet

The proposed MAFnet architecture is shown below. One video is splited into T non-overlapping clips. Then, audio and visual information are extracted with two pretrained CNNs: DenseNet [45] for visual features and VGGish [46] for audio features. The clip features are further fed into modality & temporal attention module to build a global feature containing multimodal and temporal information. This
global feature is then used to predict the label of the video. A lateral connection between visual and audio pathways is added trough the FiLM layer [44].

<p align="center">
   <img src="figs/MAFnet.png" width="400" title="MAFnet">
</p>

The trained model can be downloaded [here](https://drive.google.com/file/d/1idreklMT5dm3Z85gCdfIUMVhkjZAiGkx/view?usp=sharing).

## AVE Dataset & Features

We train and test our model on the AVE Dataset [[1]](https://sites.google.com/view/audiovisualresearch)

Audio and visual feature can be downloaded [here](https://drive.google.com/file/d/1bIiWakMw4neXDYPO_9H_gGN_UcikWa7I/view?usp=sharing). Audio feature are extracted with a VGGish network [2] and visual feature are extracted with DenseNet [3]

Scripts for generating audio and visual features are in feature_extractor folder (Feel free to modify and use it to process your audio-visual data)

## Requirements
* Python-3.6
* Tensorflow-gpu-1.15
* Keras
* Scikit-learn

* pillow
* resampy
* ffmpeg
* pickle

## Training and testing scripts

To train the network:
```
python train.py --train
```

To test the network:
```
python train.py
```

## References
[1] TIAN, Yapeng, SHI, Jing, LI, Bochen, et al. Audio-visual event localization in unconstrained videos. In : Proceedings of the European Conference on Computer Vision (ECCV). 2018. p. 247-263. [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.pdf) [Download link](https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view)

[2] HERSHEY, Shawn, CHAUDHURI, Sourish, ELLIS, Daniel PW, et al. CNN architectures for large-scale audio classification. In : IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2017. p. 131-135. [Paper](https://arxiv.org/pdf/1609.09430.pdf)

[3] HUANG, Gao, LIU, Zhuang, VAN DER MAATEN, Laurens, et al. Densely connected convolutional networks. In : Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. p. 4700-4708. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)


## Acknowledgments
Thanks to CHISTERA IGLU and the European Regional Development Fund (ERDF) for funding. 

Audio features are extracted using VGGish and visual features are extracting using DenseNet. We thank the authors for sharing their codes.