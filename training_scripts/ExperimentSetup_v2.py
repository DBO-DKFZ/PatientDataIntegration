import configs, functions_v2, dataset_classes_v1, mymodels 

import os
import torch
import time
import torchvision

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from datetime import datetime 
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

def Experiment(hparams): 
    
    print(hparams.test_flag) # indicates whether the code runs within test mode 
    # Choice of gpu 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=  hparams.gpu
    
    rp = configs.get_result_path()
    result_path = rp + hparams.experiment_id + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    
    # Experiment ID: 
    date = datetime.now().strftime("%d%m%Y")
    separator = '_'
    consider_metadata = separator.join(hparams.metadata_list)
    data = separator.join(hparams.dataset)
 
    hparams.config_id = hparams.experiment_id
    hparams.experiment_id = str(date) +'_' + hparams.experiment_id + '_'+ data + '_' + hparams.mode + '_' + hparams.encoding_scheme + '_' + consider_metadata
 
  
    print('Starting experiment ', hparams.experiment_id)
    # Device
    device = torch.device('cuda:'+hparams.gpu if torch.cuda.is_available() else 'cpu')
    print('Using device ', device)
    print('Dataset: ', hparams.dataset)
    print('Mode: ', hparams.mode)
    print('Encoding: ', hparams.encoding_scheme)
    
    # depending on dataset choose files to images and csv file:

    print('lab2')
    file_path_lab2, csv_path_lab2, result_path_lab2 = configs.get_paths('lab2', hparams.experiment_id)
    print('lab1')
    file_path_lab1, csv_path_lab1, result_path_lab1 = configs.get_paths('lab1', hparams.experiment_id)
    metadata_lab2 = pd.read_csv(csv_path_lab2)
    metadata_lab2['lab'] = np.array([0]*len(metadata_lab2['slide_name']))
    metadata_lab1 = pd.read_csv(csv_path_lab1)
    metadata_lab1['lab'] = np.array([1]*len(metadata_lab1['slide_name']))
    

    metadata = pd.DataFrame(
    { 'slide_name': metadata_lab2.slide_name.tolist() + metadata_lab1.slide_name.tolist(), 
     'label': metadata_lab2.label.tolist() + metadata_lab1.label.tolist(), 
     'age': metadata_lab2.age.tolist() + metadata_lab1.age.tolist(), 
     'gender': metadata_lab2.gender.tolist() + metadata_lab1.gender.tolist(), 
     'location': metadata_lab2.location.tolist() + metadata_lab1.location.tolist(), 
     'UV': metadata_lab2.UV.tolist() + metadata_lab1.UV.tolist(), 
     'lab': metadata_lab2.lab.tolist() + metadata_lab1.lab.tolist()   
    })

    # --------------------------
    # Handling Missing data 
    # --------------------------
    print('Preprocessing I: Handling Missing data...')
    list_of_keys = ['age', 'gender', 'location', 'label', 'lab']
    hparams.combined_metadata, hparams.labels_matrix = functions_v2.handling_missing_data(metadata, list_of_keys)
    print(hparams.combined_metadata.head())

    # Get all slides in a list 
    samples_list = []
    for sn in metadata_lab2.slide_name: 
        samples_list.append(file_path_lab2 + '/' + str(sn))
    for sn in metadata_lab1.slide_name: 
        samples_list.append(file_path_lab1 + '/' + str(sn))
    print('Dataset includes {} slides'.format(len(samples_list)))
    hparams.samples_list = np.asarray(samples_list)
    
    # --------------------------
    # Stratified data split (training, validation, test set creation)
    # --------------------------
    path_set_folds, path_set_folds_indices = functions_v2.split_samples_with_sk_multilearn_v1(hparams.splits, hparams.rs, hparams.samples_list, hparams.labels_matrix, hparams.combined_metadata,
                                                                                              result_path= result_path)
    # --------------------------
    # Encoding of the patient data  
    # --------------------------
    hparams.combined_metadata = functions_v2.encodings_v1(hparams.combined_metadata, hparams.encoding_scheme)
    print('*'*10)
    print('Check encoding...')
    print(hparams.combined_metadata.head())
    
    # --------------------------
    # Transformations  
    # --------------------------
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale = (0.7,1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.0), shear=None),
        transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=0.01),
        transforms.RandomHorizontalFlip(), # Vertical Flip? 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # --------------------------
    # Training the models for multiple runs under similar circumstances to quantify variance of the training process
    # --------------------------
    
    best_models = {}
    metric_dict = {}
    best_results = {}
    test_results = {}
    image_datasets = {}
    test_results_internal = {}
    test_results_external = {}
    
    paths = []
    fold_membership = []
    set_membership = []

    for fold in range(0, hparams.runs): 
        print('Run ', fold)
        np.random.seed(hparams.rs) # seeden 
        path_set = path_set_folds
        
        # **************************
        # Create datasets   
        # **************************
        image_datasets['train'] = dataset_classes_v1.CustomizedDataset_Fusion(path_set = path_set['train'], metadata=hparams.combined_metadata, rs_tiles = hparams.rs, nr_tiles = hparams.nr_tiles,
                                                                              transform=data_transforms['train'], non_image = hparams.metadata_list,  encoding_scheme = hparams.encoding_scheme)
        image_datasets['val'] = dataset_classes_v1.CustomizedDataset_Fusion(path_set = path_set['val'], metadata=hparams.combined_metadata, rs_tiles = hparams.rs, nr_tiles = 0,
                                                                            transform=data_transforms['val'], non_image = hparams.metadata_list,  encoding_scheme = hparams.encoding_scheme)
        image_datasets['test'] = dataset_classes_v1.CustomizedDataset_Fusion(path_set = path_set['test'], metadata=hparams.combined_metadata, rs_tiles = hparams.rs, nr_tiles = 0,
                                                                             transform=data_transforms['test'],non_image = hparams.metadata_list,  encoding_scheme = hparams.encoding_scheme)

        print('Size Trainset:', len(image_datasets['train']))
        print('Size Validationset:', len(image_datasets['val']))
        print('Size Testset:', len(image_datasets['test']))

        for x in ['train', 'val', 'test']: 
            paths.extend(image_datasets[x].samples)
            fold_membership.extend([fold]*len(image_datasets[x].samples))
            set_membership.extend([x]*len(image_datasets[x].samples))
            
        # **************************
        # Create dataloaders   
        # **************************
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers, drop_last = True),
            'val' : torch.utils.data.DataLoader(image_datasets['val'],batch_size=hparams.batch_size, shuffle= True, num_workers=hparams.num_workers, drop_last = True), 
            'test' : torch.utils.data.DataLoader(image_datasets['test'],batch_size=hparams.batch_size, shuffle= False, num_workers=hparams.num_workers,  drop_last = False)}

        # **************************
        # Model architecture    
        # **************************
        
        # setting the general pretrained feature extractor (baseline and used in all approaches)
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        
        # baseline (pure image classifier)  
        if hparams.mode == 'base': 
            print('Mode ', hparams.mode)
            for param in model.parameters():
                param.requires_grad = False
            criterion = nn.CrossEntropyLoss()
        
        # adjustment of the pretrained model: replace the last layer by a new fully connected layer 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # determine size of metadata input, necessary for initialization of the fusion models  
        if hparams.encoding_scheme == 'onehot': 
            meta_in = 0 
            if 'age' in hparams.metadata_list: 
                meta_in += 2
            if 'location' in hparams.metadata_list: 
                meta_in +=3
            if 'gender' in hparams.metadata_list: 
                meta_in += 2
            print('Encoding via onehot: meta_in parameter: ', meta_in)
        else: 
            meta_in = len(hparams.metadata_list)
        print('meta_in parameter: ', meta_in)
        
         # CAT-approach
        if hparams.mode == 'approach_1': 
            print('Mode ', hparams.mode) 
            print(meta_in, hparams.meta_out, hparams.hidden1)
            model = mymodels.FusionNet(model, meta_in = meta_in, meta_out = hparams.meta_out, hidden1 = hparams.hidden1, reduction = 1024)
            criterion = nn.CrossEntropyLoss()

        # weighted approach
        if hparams.mode == 'approach_2':
            print('Mode ', hparams.mode)
            print(meta_in, hparams.meta_out, hparams.hidden1)
            model = mymodels.FusionNet_importance(model, meta_in = meta_in, meta_out = hparams.meta_out, hidden=hparams.hidden1)
            criterion = nn.BCELoss()
        
        # SE-approach
        if hparams.mode == 'approach_3':
            print('Mode ', hparams.mode)
            hparams.meta_out = num_ftrs
            print(meta_in, hparams.meta_out, hparams.hidden1)
            model = mymodels.FusionNet_SEMul(model, meta_in = meta_in, meta_out = hparams.meta_out, hidden1=hparams.hidden1)
            criterion = nn.BCELoss() 

        model = model.to(device)

        # count trainable parameters
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Parameters to train: ', count_parameters(model)) 

        
        # **************************
        # Scheduler settings    
        # **************************
        if hparams.mode in ['approach_1','approach_2','approach_3']:
            print('First Image extractor is frozen')
            for param in model.img_extractor.parameters():
                param.requires_grad = False
            for param in model.metadata_extractor.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
            print('Optimizer works with Lr Image Extractor / Meta Extractor/ FC Layer: - (frozen) /{}/{}'.format(hparams.lr_meta_unfrozen, hparams.lr_fc_unfrozen))
            optimizer = optim.SGD([
               {'params': model.img_extractor.parameters()},
               {'params': model.metadata_extractor.parameters(), 'lr': hparams.lr_meta_unfrozen}, 
               {'params': model.fc.parameters(), 'lr': hparams.lr_meta_unfrozen} 
                ], lr=hparams.lr_frozen, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = hparams.lr_meta_unfrozen, steps_per_epoch=len(dataloaders['train']), epochs=hparams.end_epoch_p1)
        else: 
            optimizer = optim.SGD(model.parameters(), lr = hparams.lr_frozen, momentum = 0.9)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = hparams.lr_frozen, steps_per_epoch=len(dataloaders['train']), epochs=hparams.epoch_fine_tuning)
        
        print('Schedulers total steps: ', scheduler.total_steps)
        print('defined total steps ', len(dataloaders['train'])*hparams.end_epoch_p1)
        
        
        # **************************
        # Training    
        # **************************
        model, best_results, metric_dict = functions_v2.training_loop_v2(hparams.num_epochs, 
                                                                         hparams.epoch_fine_tuning, 
                                                                         hparams.lr_unfrozen, 
                                                                         optimizer,
                                                                         scheduler,
                                                                         dataloaders,
                                                                         model,
                                                                         hparams.test_flag,
                                                                         device,
                                                                         image_datasets,
                                                                         criterion,
                                                                         best_results,
                                                                         fold, 
                                                                         result_path, 
                                                                         metric_dict, 
                                                                         hparams.mode, 
                                                                         hparams)    
        # **************************
        # Evaluation on Testset  
        # **************************
        print('Performance on Testset:')
       
        test_results_internal = functions_v2.test_loop_v1(model, dataloaders, image_datasets, test_results_internal, device, fold, hparams.mode) # at the end of the training loop, always the best model is reloaded 
        best_models['fold_'+str(fold)] = {'model': model}
        

        # make sure everything is deleted 
        del model
        torch.cuda.empty_cache()
        time.sleep(10)

        print('Done.') 

 
