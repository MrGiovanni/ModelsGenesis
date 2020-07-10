<p align="center"><img width="70%" src="figures/logo.png" /></p>

--------------------------------------------------------------------------------

We have built a set of pre-trained models called <b>Generic Autodidactic Models</b>, nicknamed <b>Models Genesis</b>, because they are created <i>ex nihilo</i> (with no manual labeling), self-taught (learned by self-supervision), and generic (served as source models for generating application-specific target models). We envision that Models Genesis may serve as a primary source of transfer learning for 3D medical imaging applications, in particular, with limited annotated data. 

<p align="center"><img width="100%" src="figures/patch_generator.png" /></p>
<p align="center"><img width="85%" src="figures/framework.png" /></p>


## Paper
This repository provides the official implementation of training Models Genesis as well as the usage of the pre-trained Models Genesis in the following paper:

<b>Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis</b> <br/>
[Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup>, [Vatsal Sodha](https://github.com/vatsal-sodha)<sup>1</sup>, [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1</sup>,  <br/>
[Ruibin Feng](https://chs.asu.edu/ruibin-feng)<sup>1</sup>, [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, and [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup> <br/>
<sup>1 </sup>Arizona State University,   <sup>2 </sup>Mayo Clinic <br/>
International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2019 <br/>
<b>[Young Scientist Award](http://www.miccai.org/about-miccai/awards/young-scientist-award/)</b>  <br/>
[paper](http://www.cs.toronto.edu/~liang/Publications/ModelsGenesis/MICCAI_2019_Full.pdf) | [code](https://github.com/MrGiovanni/ModelsGenesis) | [slides](https://docs.wixstatic.com/ugd/deaea1_c5e0f8cd9cde4c3db339d866483cbcd3.pdf) | [poster](http://www.cs.toronto.edu/~liang/Publications/ModelsGenesis/Models_Genesis_Poster.pdf) | talk ([YouTube](https://youtu.be/5W_uGzBloZs), [YouKu](https://v.youku.com/v_show/id_XNDM5NjQ1ODAxMg==.html?sharefrom=iphone&sharekey=496e1494c76ed263653aa3aada61c23e6)) | [blog](https://zhuanlan.zhihu.com/p/86366534)

<b>Models Genesis</b> <br/>
[Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup>, [Vatsal Sodha](https://github.com/vatsal-sodha)<sup>1</sup>, [Jiaxuan Pang](https://github.com/MRJasonP)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, and [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup> <br/>
<sup>1 </sup>Arizona State University,   <sup>2 </sup>Mayo Clinic <br/>
Submitted to Medical Image Analysis (MedIA) for the Special Issue on MICCAI 2019 <br/>
[paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/MrGiovanni/ModelsGenesis)

## Available implementation

- keras/
- pytorch/


## Major results from our work

1. **Models Genesis outperform 3D models trained from scratch**
2. **Models Genesis top any 2D approaches, including ImageNet models and degraded 2D Models Genesis**
3. **Models Genesis (2D) offer performances equivalent to supervised pre-trained models**

The par plots presented below are produced by Matlab code in `figures/plotsuperbar.m` and the helper functions in `figures/superbar`. Credit to [superbar](https://github.com/scottclowe/superbar) by Scott Lowe.

<p align="center"><img width="90%" src="figures/results.png" /></p>

Note that learning from scratch <i>simply</i> in 3D may not necessarily yield performance better than ImageNet-based transfer learning in 2D


## Citation
If you use this code or use our pre-trained weights for your research, please cite our papers:
```
@InProceedings{zhou2019models,
  author="Zhou, Zongwei and Sodha, Vatsal and Rahman Siddiquee, Md Mahfuzur and Feng, Ruibin and Tajbakhsh, Nima and Gotway, Michael B. and Liang, Jianming",
  title="Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2019",
  year="2019",
  publisher="Springer International Publishing",
  address="Cham",
  pages="384--393",
  isbn="978-3-030-32251-9",
  url="https://link.springer.com/chapter/10.1007/978-3-030-32251-9_42"
}

@article{zhou2020models,
  title="Models Genesis",
  author="Zhou, Zongwei and Sodha, Vatsal and Pang, Jiaxuan and Gotway, Michael B and Liang, Jianming",
  journal="arXiv preprint arXiv:2004.07882",
  year="2020",
  url="https://arxiv.org/abs/2004.07882"
}
```


## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the National Institutes of Health (NIH) under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. This is a patent-pending technology.
