# Models Genesis - Incorperated with nnU-Net

By adapting Models Genesis to nnU-Net, we have so far accomplished:<br/>

&#9733; Rank # 1 in segmenting liver tumor<br/>
&#9733; Rank # 1 in segmenting hippocampus<br/>

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
Medical Image Analysis (MedIA) for the Special Issue on MICCAI 2019 <br/>
[paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/MrGiovanni/ModelsGenesis) | [slides](https://d5b3ebbb-7f8d-4011-9114-d87f4a930447.filesusr.com/ugd/deaea1_5ecdfa48836941d6ad174dcfbc925575.pdf)

## Dependencies

+ Linux
+ Python 2.7+
+ PyTorch 1.3.1

## Usage of the pre-trained nnU-Net

### 0. Before proceeding to the below steps, install nnUNet from [here](https://github.com/MIC-DKFZ/nnUNet).

### 1. Clone the repository
```bash
$ git clone https://github.com/MrGiovanni/ModelsGenesis.git
$ cd ModelsGenesis/
$ pip install -r requirements.txt
```

### 2. Download the pre-trained nnU-Net
To download the pre-trained nnU-Net, first request [here](https://www.wjx.top/jq/46747127.aspx). After submitting the form, download the pre-trained nnU-Net and save into `specify here` directory.

### 3. Modifications in order to fine-tune Models Genesis on your own target task using nnUNet framework
Models Genesis learns a general-purpose image representation that can be leveraged for a wide range of target tasks. Specifically, Models Genesis can be utilized to initialize the encoder for the target classification tasks and to initialize the encoder-decoder for the target segmentation tasks.
For loading the pre-trained Models Genesis weights in nnUNet framework, we need do to some modifications in the code.
- First add argument for weight in <ins>nnunet/training/network_training/nnUNetTrainerV2.py</ins>:
    ```python
    parser.add_argument("-w", required=False, default=None, help="Load pre-trained Models Genesis")
    ```
    then parse it using:
    ```python
    weights = args.w
    ```
    After the trainer class initialization in the same file, call **load_pretrained_weights** method to load the pre-trained Models Genesis:
    ```python
    trainer.initialize(not validation_only) #### Already exists in main method
    #### Reference for adding the call to load pre-trained weights
    #### Add below lines
    if weights != None:                                                         
        trainer.load_pretrained_weights(weights)
    ```
- Now lets define **load_pretrained_weights** inside **NetworkTrainer** class in <ins>nnunet/training/network_training/network_trainer.py</ins>:
    ```python
    def load_pretrained_weights(self,fname):                                    
        saved_model = torch.load(fname)                                         
        pretrained_dict = saved_model['state_dict']                             
        model_dict = self.network.state_dict()                                  
        # filter unnecessary keys                                               
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
    ```
### 3. Fine-tune Models Genesis for target task in nnUNet framework
Once done with the modifications, it's time to fine-tune Models Genesis for target tasks in nnUNet framework. To do so run the following command:
```bash
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD -w "pre-trained Models Genesis path"
```
For example, For 3d_fullres architecture, run below:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 TaskXXX_MYTASK FOLD -w "path to pre-trained Models Genesis"
```
Similarly it can be done for rest of the architectures.

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

@article{zhou2020models,
  title="Models Genesis",
  author="Zhou, Zongwei and Sodha, Vatsal and Pang, Jiaxuan and Gotway, Michael B and Liang, Jianming",
  journal="arXiv preprint arXiv:2004.07882",
  year="2020",
  url="https://arxiv.org/abs/2004.07882"
}
```

## Acknowledgement
We thank [Shivam Bajpai](https://github.com/sbajpai2) for his implementation of pre-trained nnU-Net. We build nnU-Net framework by referring to the released code at [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet). This is a patent-pending technology.

