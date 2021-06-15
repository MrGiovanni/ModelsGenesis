# Models Genesis - Official PyTorch Implementation

We provide the official <b>PyTorch</b> implementation of training Models Genesis as well as the usage of the pre-trained Models Genesis in the following paper:

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
Medical Image Analysis (MedIA) <br/>
<b>[MedIA Best Paper Award](http://www.miccai.org/about-miccai/awards/medical-image-analysis-best-paper-award/)</b>  <br/>
[paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/MrGiovanni/ModelsGenesis) | [slides](https://d5b3ebbb-7f8d-4011-9114-d87f4a930447.filesusr.com/ugd/deaea1_5ecdfa48836941d6ad174dcfbc925575.pdf)


## Dependencies

+ Linux
+ Python 2.7+
+ PyTorch 1.3.1


## Usage of the pre-trained Models Genesis

### 1. Clone the repository
```bash
$ git clone https://github.com/MrGiovanni/ModelsGenesis.git
$ cd ModelsGenesis/
$ pip install -r requirements.txt
```

### 2. Download the pre-trained Models Genesis
To download the pre-trained Models Genesis, first request [here](https://www.wjx.top/jq/46747127.aspx). 
After submitting the form, download the pre-trained Genesis Chest CT and save into `./pretrained_weights/Genesis_Chest_CT.pt` directory.


### 3. Fine-tune Models Genesis on your own target task
Models Genesis learn a general-purpose image representation that can be leveraged for a wide range of target tasks. Specifically, Models Genesis can be utilized to initialize the encoder for the target <i>classification</i> tasks and to initialize the encoder-decoder for the target <i>segmentation</i> tasks.

As for the target classification tasks, the 3D deep model can be initialized with the pre-trained encoder using the following example:
```python


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d

# prepare your own data
train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2( F.relu(self.linear_out))
        return final_out
        
base_model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
target_model = TargetNet(base_model)
target_model.to(device)
target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    scheduler.step(epoch)
    target_model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = F.sigmoid(target_model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

As for the target segmentation tasks, the 3D deep model can be initialized with the pre-trained encoder-decoder using the following example:
```python

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model

model = unet3d.UNet3D()

#Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```

**Prepare your own data:** If the image modality in your target task is CT, we suggest that all the intensity values be clipped on the min (-1000) and max (+1000) interesting Hounsfield Unit range and then scale between 0 and 1. If the image modality is MRI, we suggest that all the intensity values be clipped on min (0) and max (+4000) interesting range and then scale between 0 and 1. For any other modalities, you may want to first clip on the meaningful intensity range and then scale between 0 and 1. 

We adopt input cubes shaped in (N, 1, 64, 64, 32) during model pre-training, where N denotes the number of training samples。 When fine-tuning the pre-trained Models Genesis, **any arbitrary input size** is acceptable as long as it is divisible by 16 (=2^4) due to four down-sampling layers in V-Net. That said, to segment larger objects, such as liver, kidney, or big nodule :-( you may want to try
```python
input_channels, input_rows, input_cols, input_deps = 1, 128, 128, 64
input_channels, input_rows, input_cols, input_deps = 1, 160, 160, 96
```
or even larger input size as you wish.


## Learn Models Genesis from your own unlabeled data

### 1. Clone the repository
```bash
$ git clone https://github.com/MrGiovanni/ModelsGenesis.git
$ cd ModelsGenesis/
$ pip install -r requirements.txt
```

### 2. Create the data generator (LUNA-2016 for example)

#### For your convenience, we have provided our own extracted 3D cubes from LUNA16. 

Download from [Google Drive](https://drive.google.com/drive/folders/1ZEQHc4FLFHK204UJ1iABQVrjQOFmx_gX?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/1U4qzEu-Ijl8hkSCLTr-agA) <- code: m8g4. Each sub-folder is named as 'bat_N_s_64x64x32', where *N* denotes the number of cubes extracted from each patient. You may select the scale of training samples accordingly based on your resources in hand: larger *N* demands longer learning time and more powerful GPUs/CPUs, while may (or may not) result in a more generic visual representation. We have adopted *N*=32 in our MICCAI paper. 

- The processed cubes directory structure
```
generated_cubes/
    |--  bat_32_s_64x64x32_0.npy: cubes extracted from subset0 in luna16
    |--  bat_32_s_64x64x32_1.npy: cubes extracted from subset1 in luna16
    |--  bat_32_s_64x64x32_2.npy: cubes extracted from subset2 in luna16
    |--  bat_32_s_64x64x32_3.npy: cubes extracted from subset3 in luna16
    |--  bat_32_s_64x64x32_4.npy: cubes extracted from subset4 in luna16
    |--  bat_32_s_64x64x32_5.npy: cubes extracted from subset5 in luna16
    |--  bat_32_s_64x64x32_6.npy: cubes extracted from subset6 in luna16
    |--  bat_32_s_64x64x32_7.npy: cubes extracted from subset7 in luna16
    |--  bat_32_s_64x64x32_8.npy: cubes extracted from subset8 in luna16
    |--  bat_32_s_64x64x32_9.npy: cubes extracted from subset9 in luna16
```

#### You can also extract 3D cubes by your own following two steps below:

**Step 1**: Download LUNA-2016 dataset from the challenge website (https://luna16.grand-challenge.org/download/) and save to `./datasets/luna16` directory.

**Step 2**: Extract 3D cubes from the patient data by running the script below. The extracted 3D cubes will be saved into `./generated_cubes` directory.
```bash
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data datasets/luna16 \
--save generated_cubes
done
```

### 3. Pre-train Models Genesis (LUNA-2016 for example)
```bash
python -W ignore pytorch/Genesis_Chest_CT.py
```
Your pre-trained Models Genesis will be saved at `./pytorch/pretrained_weights/Vnet-genesis_chest_ct.pt`.


## Citation
If you use this code or use our pre-trained weights for your research, please cite our [paper](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_42):
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

@article{zhou2021models,
  title="Models Genesis",
  author="Zhou, Zongwei and Sodha, Vatsal and Pang, Jiaxuan and Gotway, Michael B and Liang, Jianming",
  journal="Medical Image Analysis",
  volume = "67",
  pages = "101840",
  year = "2021",
  issn = "1361-8415",
  doi = "https://doi.org/10.1016/j.media.2020.101840",
  url = "http://www.sciencedirect.com/science/article/pii/S1361841520302048",
}

@phdthesis{zhou2021towards,
  title={Towards Annotation-Efficient Deep Learning for Computer-Aided Diagnosis},
  author={Zhou, Zongwei},
  year={2021},
  school={Arizona State University}
}
```


## Acknowledgement
We thank [Jiaxuan Pang](https://github.com/MRJasonP) and [Vatsal Sodha](https://github.com/vatsal-sodha) for their implementation of Models Genesis in PyTorch. We build 3D U-Net architecture by referring to the released code at [mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch). This is a patent-pending technology.
