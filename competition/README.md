# Models Genesis - Incorperated with nnU-Net

By adapting Models Genesis to nnU-Net, we (user: JLiangLab) have so far accomplished:<br/>

&#9733; Rank # 1 in segmenting liver tumor<br/>
&#9733; Outperform scratch nnU-Net in segmenting lung nodule<br/>

In this page, we provide the pre-trained 3D nnU-Net and describe the usage of the model. The original idea has been presented in the following papers:

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

<b>Pre-Trained Models for nnUNet</b> <br/>
[Shivam Bajpai](https://www.linkedin.com/in/shivam-bajpai-69937734/) <br/>
Master's Degree Thesis, Arizona State University <br/>
[paper](https://www.proquest.com/docview/2532501890?pq-origsite=gscholar&fromopenview=true) | [code](https://github.com/MrGiovanni/ModelsGenesis/tree/master/competition)

## Dependencies

+ Linux
+ Python 3.7+
+ PyTorch 1.6+

## Featured results

| Experiment      | Liver 1_Dice (val) | Liver 2_Dice (val) | Liver 1_Dice (test) | Liver 2_Dice (test) |
|-----------------|:------------------:|:------------------:|:-----------------:|:-----------------:|
| nnU-Net (reported)               | 95.71 | 63.72 |  95.75 | 75.97 |
| nnU-Net (reproduced)          | 95.74 ± 0.34 | 62.61 ± 0.51 | - | - |
| Pre-trained nnU-Net   | 96.16 ± 0.02 | 64.47 ± 0.45 | 95.72 | 77.50 |

| Experiment      | Lung 1_Dice (val) | Lung 1_Dice (test) |
|-----------------|:-----------------:|:----------------:|
| nnU-Net (reported)               | 72.11 | 73.97 |
| nnU-Net (reproduced)          | 69.50 ± 1.13 | - |
| Pre-trained nnU-Net   | 71.80 ± 1.40 | 74.54 |

Note: Since the network architectures vary from task to task, we select two tasks, i.e., Task003_Liver and Task006_Lung, which share exactly the same architecture. One of the typical transfer learning barriers is that the proxy and target tasks' architectures must be the same; otherwise, the pre-trained weights will be meaningless. Pre-training unique architectures for every individual task would take too much computational power and time to accomplish. It would be great to address this limitation for Models Genesis, and for transfer learning in general.

## Usage of the pre-trained nnU-Net (Task003_Liver as an example)


### 0. Before proceeding to the below steps, install nnUNet from [here](https://github.com/MIC-DKFZ/nnUNet).

- Create virtual environment. [Here is a quick how-to for Ubuntu](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
- Install [PyTorch](https://pytorch.org/get-started/locally/)
- Install nnU-Net as below
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install git+https://github.com/MIC-DKFZ/batchgenerators.git
pip install -e .
```
- Set a few environment variables. Please follow the instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md)


### 1. Download the pre-trained nnU-Net
To download the pre-trained nnU-Net, first request [here](https://www.wjx.top/jq/46747127.aspx). After submitting the form, download the pre-trained nnU-Net `genesis_nnunet_luna16_006.model`, create a new folder `pretrained_weights`, and save the model to the `pretrained_weights/` directory.

### 2. Modify codes in two files


- Modify ```nnunet/run/run_training.py```:
```python
# Add an argument for pre-trained weights
parser.add_argument("-w", required=False, default=None, help="Load pre-trained Models Genesis") 
...
# Existing in the file
args = parser.parse_args() 
...
# Parse it to variable "weights"
weights = args.w 
...
# Existing in the file
trainer.initialize(not validation_only) 
# Add below lines
if weights != None:                                                         
    trainer.load_pretrained_weights(weights)
```

- Add a new function under the "NetworkTrainer" class in ```nnunet/training/network_training/network_trainer.py```:
```python
def load_pretrained_weights(self,fname):                                    
    saved_model = torch.load(fname)                                         
    pretrained_dict = saved_model['state_dict']                             
    model_dict = self.network.state_dict()                                  
    fine_tune = True                                                        
    for key, _ in model_dict.items():                                       
       if ('conv_blocks' in key):                                           
           if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
               continue                                                     
           else:                                                            
               fine_tune = False                                            
               break                                                        
    # filter unnecessary keys                                               
    if fine_tune:                                                           
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if      
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict                       
        model_dict.update(pretrained_dict)                                  
        # print(model_dict)                                                     
        print("############################################### Loading pre-trained Models Genesis from ",fname)
        print("Below is the list of overlapping blocks in pre-trained Models Genesis and nnUNet architecture:")
        for key, _ in pretrained_dict.items():                              
            print(key)                                                      
        print("############################################### Done")                                                            
        self.network.load_state_dict(model_dict)                            
    else:                                                                   
        print('############################################### Training from scratch')```
```

### 3. Fine-tune the pre-trained nnU-Net

- Run the following command to fine-tune the model on Task003_Liver:

```bash
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres nnUNetTrainerV2 Task003_Liver $FOLD -w pretrained_weights/genesis_nnunet_luna16_006.model
done
```

To fine-tune nnU-Net, the general structure of the command is:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD -w pretrained_weights/genesis_nnunet_luna16_006.model
```


## Pre-train nnU-Net from your own dataset

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

### 3. Pre-train Models Genesis for nnU-Net (LUNA-2016 for example)

```bash
cd competition/
python -W ignore Genesis_nnUNet.py
```
Your pre-trained nnU-Net will be saved at `./competition/pretrained_weights/genesis_nnunet_luna16_006.model`.

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

@phdthesis{bajpai2021pre,
  title={Pre-Trained Models for nnUNet},
  author={Bajpai, Shivam},
  year={2021},
  school={Arizona State University}
}

@phdthesis{zhou2021towards,
  title={Towards Annotation-Efficient Deep Learning for Computer-Aided Diagnosis},
  author={Zhou, Zongwei},
  year={2021},
  school={Arizona State University}
}
```

## Acknowledgement
We thank [Shivam Bajpai](https://github.com/sbajpai2) and [Vatsal Sodha](https://github.com/vatsal-sodha) for their implementation of pre-trained nnU-Net. We build nnU-Net framework by referring to the released code at [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). This is a patent-pending technology.

