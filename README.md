<p align="center"><img width="50%" src="figures/logo.png" /></p>

--------------------------------------------------------------------------------
We have built a set of pre-trained models called <b>Generic Autodidactic Models</b>, nicknamed <b>Models Genesis</b>, because they are created <i>ex nihilo</i> (with no manual labeling), self-taught (learned by self-supervision), and generic (served as source models for generating application-specific target models). This repository provides a Keras implementation of training Models Genesis as well as the pre-trained Genesis Chest CT. We envision that Models Genesis may serve as a primary source of transfer learning for 3D medical imaging applications, in particular, with limited annotated data. 

<p align="center"><img width="100%" src="figures/patch_generator.png" /></p>
<p align="center"><img width="80%" src="figures/framework.png" /></p>

<br/>

## Paper
[Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis](http://www.cs.toronto.edu/~liang/Publications/ModelsGenesis/MICCAI_2019_Full.pdf) <br/>
[Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup>, [Vatsal Sodha](https://github.com/vatsal-sodha)<sup>2</sup>, [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1</sup>,  <br/>
[Ruibin Feng](https://chs.asu.edu/ruibin-feng)<sup>1</sup>, [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/)<sup>1</sup>, [Michael Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup> and [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup> <br/>
<sup>1 </sup>Arizona State University,   <sup>2 </sup>Mayo Clinic <br/>
International Conference on Medical Image Computing and Computer Assisted Intervention ([MICCAI](https://www.miccai2019.org/)), 2019 (<b>Oral</b>)

<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/yunjey/StarGAN.git
$ cd ModelsGenesis/
```

<br/>

## Model Zoo
Download the pre-trained Models Genesis with the following script
```bash
bash ./pretrained_models/download_model.sh model_name
```
The models will be automatically saved to `./checkpoints/model_name/latest_net_G.t7`.

<br/>

## Citation
If you use this code or use our pre-trained weights for your research, please cite our [paper](http://www.cs.toronto.edu/~liang/Publications/ModelsGenesis/MICCAI_2019_Full.pdf):
```
@incollection{zhou2019models,
author = {Zhou, Zongwei and Sodha, Vatsal and Rahman Siddiquee, Md Mahfuzur and Feng, Ruibin and Tajbakhsh, Nima and Gotway, Michael and Liang, Jianming},
title = {Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis},
booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
month = {October},
year = {2019}
}
```

<br/>

## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH.
