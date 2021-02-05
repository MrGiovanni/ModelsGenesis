import os
import shutil
import csv
import random

class bms_config:
    arch = 'Vnet'
    
    # data
    data = '/mnt/dataset/shared/zongwei/BraTS'
    csv = "data/bms"
    deltr = 30
    input_rows = 64 
    input_cols = 64
    input_deps = 32
    crop_rows = 100
    crop_cols = 100
    crop_deps = 50
    
    # model
    optimizer = 'adam'
    lr = 1e-3
    patience = 30
    verbose = 1
    batch_size = 16
    workers = 1
    max_queue_size = workers * 1
    nb_epoch = 10000
    
    def __init__(self, args):
        self.exp_name = self.arch + '-' + args.suffix
        if args.data is not None:
            self.data = args.data
        
        if args.suffix == 'random':
            self.weights = None
        elif args.suffix == 'genesis':
            self.weights = 'pretrained_weights/Genesis_Chest_CT.h5'
        elif args.suffix == 'genesis-autoencoder':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-autoencoder.h5'
        elif args.suffix == 'genesis-nonlinear':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-nonlinear.h5'
        elif args.suffix == 'genesis-localshuffling':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-localshuffling.h5'
        elif args.suffix == 'genesis-outpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-outpainting.h5'
        elif args.suffix == 'genesis-inpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-inpainting.h5'
        elif args.suffix == 'denoisy':
            self.weights = 'pretrained_weights/denoisy.h5'
        elif args.suffix == 'patchshuffling':
            self.weights = 'pretrained_weights/patchshuffling.h5'
        elif args.suffix == 'hg':
            self.weights = 'pretrained_weights/hg.h5'
        else:
            raise
        
        train_ids = self._load_csv(os.path.join(self.csv, "fold_1.csv")) + self._load_csv(os.path.join(self.csv, "fold_2.csv"))
        random.Random(4).shuffle(train_ids)
        self.validation_ids = train_ids[:len(train_ids) // 8]
        self.train_ids = train_ids[len(train_ids) // 8:]
        self.test_ids = self._load_csv(os.path.join(self.csv, "fold_3.csv"))
        self.num_train = len(self.train_ids)
        self.num_validation = len(self.validation_ids)
        self.num_test = len(self.test_ids)
        
        # logs
        self.model_path = os.path.join("models/bms", "run_"+str(args.run))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
            
    def _load_csv(self, foldfile=None):
        assert foldfile is not None
        patient_ids = []
        with open(foldfile, 'r') as f:
            reader = csv.reader(f, lineterminator='\n')
            patient_ids.extend(reader)
        for i, item in enumerate(patient_ids):
            patient_ids[i] = item[0]
        return patient_ids
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_ids' in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
class ecc_config:
    arch = 'Vnet'
    
    # data
    data = '/mnt/dfs/zongwei/Academic/MICCAI2020/Genesis_PE/dataset/augdata/VOIR'
    csv = "data/ecc"
    clip_min = -1000
    clip_max = 1000
    input_rows = 64
    input_cols = 64
    input_deps = 64
    
    # model
    optimizer = 'adam'
    lr = 1e-3
    patience = 38
    verbose = 1
    batch_size = 24
    workers = 1
    max_queue_size = workers * 1
    nb_epoch = 10000
    num_classes = 1
    verbose = 1
    
    def __init__(self, args=None):
        self.exp_name = self.arch + '-' + args.suffix + '-cv-' + str(args.cv)
        if args.data is not None:
            self.data = args.data
            
        if args.suffix == 'random':
            self.weights = None
        elif args.suffix == 'genesis':
            self.weights = 'pretrained_weights/Genesis_Chest_CT.h5'
        elif args.suffix == 'genesis-autoencoder':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-autoencoder.h5'
        elif args.suffix == 'genesis-nonlinear':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-nonlinear.h5'
        elif args.suffix == 'genesis-localshuffling':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-localshuffling.h5'
        elif args.suffix == 'genesis-outpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-outpainting.h5'
        elif args.suffix == 'genesis-inpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-inpainting.h5'
        elif args.suffix == 'denoisy':
            self.weights = 'pretrained_weights/denoisy.h5'
        elif args.suffix == 'patchshuffling':
            self.weights = 'pretrained_weights/patchshuffling.h5'
        elif args.suffix == 'hg':
            self.weights = 'pretrained_weights/hg.h5'
        else:
            raise
            
        # logs
        assert args.subsetting is not None
        self.model_path = os.path.join("models/ecc", "run_"+str(args.run), args.subsetting)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        self.patch_csv_path = 'Patch-20mm-cv-'+str(args.cv)+'-features_output_2_iter-100000.csv'
        self.candidate_csv_path = 'Candidate-20mm-cv-'+str(args.cv)+'-features_output_2_iter-100000.csv'
        self.csv_froc = 'features_output_2_iter-100000.csv'

    def display(self):
        print("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self,a)):
                print("{:30} {}".format(a,getattr(self,a)))
                #print("\n")
                
class ncc_config:
    arch = 'Vnet'
    
    # data
    data = '/mnt/dataset/shared/zongwei/LUNA16/LUNA16_FPR_32x32x32'
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_min = -1000
    hu_max = 1000
    input_rows = 64
    input_cols = 64
    input_deps = 32
    
    # model
    optimizer = 'adam'
    lr = 1e-3
    patience = 10
    verbose = 1
    batch_size = 24
    workers = 1
    max_queue_size = workers * 1
    nb_epoch = 10000
    num_classes = 1
    verbose = 1
    
    def __init__(self, args=None):
        self.exp_name = self.arch + '-' + args.suffix
        if args.data is not None:
            self.data = args.data
            
        if args.suffix == 'random':
            self.weights = None
        elif args.suffix == 'genesis':
            self.weights = 'pretrained_weights/Genesis_Chest_CT.h5'
        elif args.suffix == 'genesis-autoencoder':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-autoencoder.h5'
        elif args.suffix == 'genesis-nonlinear':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-nonlinear.h5'
        elif args.suffix == 'genesis-localshuffling':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-localshuffling.h5'
        elif args.suffix == 'genesis-outpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-outpainting.h5'
        elif args.suffix == 'genesis-inpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-inpainting.h5'
        elif args.suffix == 'denoisy':
            self.weights = 'pretrained_weights/denoisy.h5'
        elif args.suffix == 'patchshuffling':
            self.weights = 'pretrained_weights/patchshuffling.h5'
        elif args.suffix == 'hg':
            self.weights = 'pretrained_weights/hg.h5'
        else:
            raise
            
        # logs
        self.model_path = os.path.join("models/ncc", "run_"+str(args.run))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    def display(self):
        print("Configurations")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self,a)):
                print("{:30} {}".format(a,getattr(self,a)))
                #print("\n")

class ncs_config:
    arch = 'Vnet'
    
    # data
    data = '/mnt/dataset/shared/zongwei/LIDC'
    input_rows = 64 
    input_cols = 64
    input_deps = 32
    
    # model
    optimizer = 'adam'
    lr = 1e-3
    patience = 50
    verbose = 1
    batch_size = 16
    workers = 1
    max_queue_size = workers * 1
    nb_epoch = 10000
    
    def __init__(self, args):
        self.exp_name = self.arch + '-' + args.suffix
        if args.data is not None:
            self.data = args.data
        
        if args.suffix == 'random':
            self.weights = None
        elif args.suffix == 'genesis':
            self.weights = 'pretrained_weights/Genesis_Chest_CT.h5'
        elif args.suffix == 'genesis-autoencoder':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-autoencoder.h5'
        elif args.suffix == 'genesis-nonlinear':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-nonlinear.h5'
        elif args.suffix == 'genesis-localshuffling':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-localshuffling.h5'
        elif args.suffix == 'genesis-outpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-outpainting.h5'
        elif args.suffix == 'genesis-inpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-inpainting.h5'
        elif args.suffix == 'denoisy':
            self.weights = 'pretrained_weights/denoisy.h5'
        elif args.suffix == 'patchshuffling':
            self.weights = 'pretrained_weights/patchshuffling.h5'
        elif args.suffix == 'hg':
            self.weights = 'pretrained_weights/hg.h5'
        else:
            raise
        
        # logs
        self.model_path = os.path.join("models/ncs", "run_"+str(args.run))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
        
class lcs_config:
    arch = 'Vnet'
    
    # data
    data = '/mnt/dfs/zongwei/Academic/MICCAI2019/Data/LiTS/3D_LiTS_NPY_256x256xZ'
    nii = '/mnt/dataset/shared/zongwei/LiTS/Tr'
    obj = 'liver'
    train_idx = [n for n in range(0, 100)]
    valid_idx = [n for n in range(100,  115)]
    test_idx = [n for n in range(115, 130)]
    num_train = len(train_idx)
    num_valid = len(valid_idx)
    num_test = len(test_idx)
    hu_max = 1000
    hu_min = -1000
    input_rows = 64
    input_cols = 64
    input_deps = 32
    
    # model
    optimizer = 'adam'
    lr = 1e-2
    patience = 20
    verbose = 1
    batch_size = 16
    workers = 1
    max_queue_size = workers * 1
    nb_epoch = 10000
    
    def __init__(self, args):
        self.exp_name = self.arch + '-' + args.suffix
        if args.data is not None:
            self.data = args.data
        
        if args.suffix == 'random':
            self.weights = None
        elif args.suffix == 'genesis':
            self.weights = 'pretrained_weights/Genesis_Chest_CT.h5'
        elif args.suffix == 'genesis-autoencoder':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-autoencoder.h5'
        elif args.suffix == 'genesis-nonlinear':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-nonlinear.h5'
        elif args.suffix == 'genesis-localshuffling':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-localshuffling.h5'
        elif args.suffix == 'genesis-outpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-outpainting.h5'
        elif args.suffix == 'genesis-inpainting':
            self.weights = 'pretrained_weights/Genesis_Chest_CT-inpainting.h5'
        elif args.suffix == 'denoisy':
            self.weights = 'pretrained_weights/denoisy.h5'
        elif args.suffix == 'patchshuffling':
            self.weights = 'pretrained_weights/patchshuffling.h5'
        elif args.suffix == 'hg':
            self.weights = 'pretrained_weights/hg.h5'
        else:
            raise
        
        # logs
        self.model_path = os.path.join("models/lcs", "run_"+str(args.run))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.logs_path = os.path.join(self.model_path, "logs")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not '_idx' in a:
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
