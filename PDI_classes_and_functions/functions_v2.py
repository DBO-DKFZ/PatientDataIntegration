import os
import numpy as np
import pandas as pd
import time
import copy
import math
import random 

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import torch.optim as optim
from torch.optim import lr_scheduler

from datetime import datetime 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

   
def handling_missing_data(dataframe, list_of_keys): 
    "find all 'nans' within the matrix and replace them by the mean (for continuous) or the most common value (for categorical features)"
    labels_matrix = np.asarray(dataframe[list_of_keys])
    for counter, key in enumerate(list_of_keys): 
        print(key)
        missing_indices = dataframe[key][pd.isnull(dataframe[key])==True].index
        print('number of missing data: ', len(missing_indices))
        if len(missing_indices) != 0:
            if key == 'age': 
                age_mean = np.mean(dataframe[key][pd.isnull(dataframe[key])==False])
                print('For age, replace by the mean age: ', age_mean)
                labels_matrix[missing_indices, counter] = age_mean
                dataframe.loc[missing_indices, key] = age_mean
                #print(dataframe.loc[missing_indices,'slide_name'])
            if key == 'location':
                #print(len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1]) , len(dataframe[key][dataframe[key] == 2]))
                loc = np.argmax([len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1]) , len(dataframe[key][dataframe[key] == 2])])
                print('For location, replace by the most common location: ', loc)
                labels_matrix[missing_indices, counter] = loc
                dataframe.loc[missing_indices, key] = loc
                #print(dataframe.loc[missing_indices,'slide_name'])
            if key == 'gender':
               # print(len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1]))
                sex = np.argmax([len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1])]) 
                print('For gender, replace by the most common gender: ', sex)
                labels_matrix[missing_indices, counter] = sex 
                dataframe.loc[missing_indices, key] = sex
                #print(dataframe.loc[missing_indices,'slide_name'])
            if key == 'UV':
                #print(len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1]))
                uv = np.argmax([len(dataframe[key][dataframe[key] == 0]),len(dataframe[key][dataframe[key] == 1])])
                print('For location, replace by the most common location: ',  uv)
                labels_matrix[missing_indices, counter] = uv
                dataframe.loc[missing_indices, key] = uv
                #print(dataframe.loc[missing_indices,'slide_name'])
    return dataframe, labels_matrix

def split_samples_with_sk_multilearn_v1(n_folds, rs, samples_list, labels_matrix, metadata, result_path=None):
    """
    Parameters: 
    n_folds: number of folds
    rs: random seed for reproducing
    samples_list: all samples in a list (slides)
    labels_matrix: all data for stratification
    metadata: necessary for plotting the distribution
    --------------------------------------------------
    Returns: 
    path_set_folds: dictionary, giving for each fold (fold_0 ... fold_x) and set ('train'/'val'/'test') a list of slide paths 
    path_set_folds_indices: same, but returns a list of indices for sample_list 
    """
    
    np.random.seed(rs)
    splits = int(n_folds)
    k_fold = IterativeStratification(n_splits=splits, order=1) 
    folds = [x[1] for x in k_fold.split(samples_list, labels_matrix)] 
    
    print('Stratified Folds (Checkpoint: Equally distributed?): ')
    for counter, fold in enumerate(folds):
        print('Fold {}:label nevi {}, label mel {}, Age mean: {}, Females: {}, Males: {}, Extremität: {}, Kopf: {}, Rumpf: {}, Lab 0: {}, Lab 1: {}'.format(counter,
        len(np.where(metadata['label'][fold]==0)[0]),
        len(np.where(metadata['label'][fold]==1)[0]), 
        metadata['age'][fold].mean(), 
        len(np.where(metadata['gender'][fold]==0)[0]),
        len(np.where(metadata['gender'][fold]==1)[0]), 
        len(np.where(metadata['location'][fold]==0)[0]),
        len(np.where(metadata['location'][fold]==1)[0]),
        len(np.where(metadata['location'][fold]==2)[0]), 
        len(np.where(metadata['lab'][fold]==0)[0]), 
        len(np.where(metadata['lab'][fold]==1)[0])))
    print('*'*40)
    
    all_folds = [samples_list[x] for x in folds]  
    folds_variants = np.arange(0,splits)
    images = [0]
    listOfFolds =[0]
    listOfSets = [0]
        
    # validation, testing, training
    splitting = [folds_variants[0:2], folds_variants[2:6], folds_variants[6:]]
    # validation 
    val_fold = np.concatenate(([all_folds[idx] for idx in splitting[0]]))
    val_fold_indices = np.concatenate(([folds[idx] for idx in splitting[0]]))
    # testing 
    test_fold = np.concatenate(([all_folds[idx] for idx in splitting[1]]))
    test_fold_indices = np.concatenate(([folds[idx] for idx in splitting[1]]))
    # training 
    train_fold = np.concatenate(([all_folds[idx] for idx in splitting[2]]))
    train_fold_indices = np.concatenate(([folds[idx] for idx in splitting[2]]))

    path_set_folds = {'train': train_fold, 'val': val_fold, 'test': test_fold}
    path_set_folds_indices = {'train': train_fold_indices, 'val': val_fold_indices, 'test': test_fold_indices}

    images.extend(train_fold)
    images.extend(val_fold)
    images.extend(test_fold)
    listOfFolds.extend([fold]*(len(train_fold)+len(val_fold)+len(test_fold)))
    listOfSets.extend(['train']*(len(train_fold)))
    listOfSets.extend(['val']*(len(val_fold)))
    listOfSets.extend(['test']*(len(test_fold)))

    
    print('#Slides in Trainingset: {} ({} Nevi/{} Mels)'.format(len(train_fold), len(np.where(metadata['label'][train_fold_indices] == 0)[0]), len(np.where(metadata['label'][train_fold_indices] == 1)[0])))
    print('#Slides in Validationset:: {} ({} Nevi/{} Mels)'.format(len(val_fold), len(np.where(metadata['label'][val_fold_indices] == 0)[0]), len(np.where(metadata['label'][val_fold_indices] == 1)[0])))
    print('#Slides in Testset: {} ({} Nevi/{} Mels)'.format(len(test_fold), len(np.where(metadata['label'][test_fold_indices] == 0)[0]), len(np.where(metadata['label'][test_fold_indices] == 1)[0])))
    
    # save folds in a Dataframe 
    if result_path != None: 
        df = pd.DataFrame({"Fold": listOfFolds, 'Images': images, 'Set': listOfSets})
        path_saving = result_path + 'folds.csv'
        if not os.path.exists(result_path):
                os.makedirs(result_path)
        df.to_csv(path_saving, index = False)
        
    return path_set_folds, path_set_folds_indices

def encodings_v1(combined_metadata, encoding_scheme):
    """
    Encodes the patient data depending on the selected encoding schemes
    implemented encoding schemes are:
        scale01: scales all parameters in the range of 0 and 1
        onehot: one hot encoding
        unscaled: no encoding is applied, data is directly taken from the original dataframe 
        classified_2: Age is devided into two classes: older/younger than 60; remaining parameters are not changed 
        classified_3: Age is devided into three classes: younger than 30, between 30 and 60, older than 60; remaining parameters are not changed 
    ----------------
    returns a dataframe including the encoded patient data 
    """
    if encoding_scheme == 'scale01':
        age = np.asarray(combined_metadata['age']).reshape((-1,1))
        loc = np.asarray(combined_metadata['location']).reshape((-1,1))
        scaler = MinMaxScaler()
        scaler.fit(age)
        age_norm = scaler.transform(age)
        scaler.fit(loc)
        loc_norm = scaler.transform(loc)
        combined_metadata['age'] = age_norm
        combined_metadata['location'] = loc_norm
        
    if encoding_scheme == 'onehot': 
        # age 
        age = []
        for a in combined_metadata.age: 
            if a <= 60:
                age.append([1,0])
            else:
                age.append([0,1]) 
        # gender
        gender = []
        for g in combined_metadata.gender: 
            if g == 0:
                gender.append([1,0])
            else: 
                gender.append([0,1])
                
        # location
        loc = []
        for l in combined_metadata.location: 
            if l == 0:
                loc.append([1,0,0])
            elif l == 1: 
                loc.append([0,1,0])
            else:
                loc.append([0,0,1]) 
                
        combined_metadata.age = age
        combined_metadata.gender = gender
        combined_metadata.location = loc 
    
    if encoding_scheme == 'unscaled':
        print('Nothing happens')
        
    if encoding_scheme == 'classified_3': 
        #age in three classes: 
        age = []
        for a in combined_metadata.age: 
            if a <= 30:
                age.append(int(0))
            elif a > 60: 
                age.append(int(2))
            else:
                age.append(int(1))
        combined_metadata.age = age
    
    if encoding_scheme == 'classified_2': 
        #age in two classes: 
        age = []
        for a in combined_metadata.age: 
            if a <= 60:
                age.append(int(0))
            else:
                age.append(int(1))
        combined_metadata.age = age
        
    return combined_metadata

def training_loop_v2(num_epochs, epoch_fine_tuning, lr_unfrozen, optimizer, scheduler, dataloaders, model, test_flag, device, image_datasets, criterion, best_results, fold, result_path, metric_dict, mode, hparams):
    since = time.time()
    
    lr = []
    
    trainings_loss = []
    validation_loss = []
    
    balanced_accuracy_tiles_train = []
    balanced_accuracy_slides_train = []
    balanced_accuracy_tiles_val = []
    balanced_accuracy_slides_val = []
    
    best_bal_acc = 0 # tile level 
    epoch_best = 0
    lowest_validation_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    
    scheduler_counter = 0 
    for epoch in tqdm(range(num_epochs)):
        if hparams.mode in ['approach_1','approach_2','approach_3']:
            if epoch == hparams.end_epoch_p1:
                scheduler_counter = 0
                print('First Image extractor is frozen')
                for param in model.img_extractor.parameters():
                    param.requires_grad = True
                for param in model.metadata_extractor.parameters():
                    param.requires_grad = False
                for param in model.fc.parameters():
                    param.requires_grad = True
                print('Optimizer works with Lr Image Extractor / Meta Extractor/ FC Layer: {}/- (frozen)/{}'.format(hparams.lr_unfrozen, hparams.lr_fc_unfrozen))
                optimizer = optim.SGD([
                   {'params': model.img_extractor.parameters()},
                   {'params': model.metadata_extractor.parameters(), 'lr': hparams.lr_meta_unfrozen}, 
                   {'params': model.fc.parameters(), 'lr': hparams.lr_unfrozen} 
                    ], lr=hparams.lr_unfrozen, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = hparams.lr_unfrozen, steps_per_epoch = len(dataloaders['train']), epochs = int(hparams.end_epoch_p2 - hparams.end_epoch_p1))
                print('Schedulers total steps: ', scheduler.total_steps)
                print('defined total steps ', len(dataloaders['train'])* int(hparams.end_epoch_p2 - hparams.end_epoch_p1))
            if epoch == hparams.end_epoch_p2:
                scheduler_counter = 0
                print('First Image extractor is frozen')
                for param in model.img_extractor.parameters():
                    param.requires_grad = True
                for param in model.metadata_extractor.parameters():
                    param.requires_grad = True
                for param in model.fc.parameters():
                    param.requires_grad = True
                print('Optimizer works with Lr Image Extractor / Meta Extractor/ FC Layer: {}/{}/{}'.format( hparams.lr_fine,  hparams.lr_fine, hparams.lr_fine))
                optimizer = optim.SGD([
                   {'params': model.img_extractor.parameters()},
                   {'params': model.metadata_extractor.parameters(), 'lr': hparams.lr_fine}, 
                   {'params': model.fc.parameters(), 'lr': hparams.lr_fine} 
                    ], lr=hparams.lr_fine, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = hparams.lr_fine, steps_per_epoch = len(dataloaders['train']), epochs = int(num_epochs - (hparams.end_epoch_p2)))
                print('Schedulers total steps: ', scheduler.total_steps)
                print('defined total steps ', len(dataloaders['train'])* int(num_epochs - (hparams.end_epoch_p2 + hparams.end_epoch_p1)))
        else: 
            if epoch == hparams.epoch_fine_tuning: 
                optimizer = optim.SGD(model.parameters(), lr = hparams.lr_unfrozen, momentum = 0.9)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = hparams.lr_unfrozen, steps_per_epoch = len(dataloaders['train']), epochs = int(num_epochs - epoch_fine_tuning))
                for param in model.parameters():
                    param.requires_grad = True

            
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        
        for phase in ['train', 'val']: 
            if phase == 'train': 
                model.train()
            else: 
                model.eval()
        
            running_loss = 0.0 
            
            y_preds = []
            y_trues = []
            y_scores = []
            slide_id_list = []
           
            print('Scheduler_counter before a new run through the dataloader: ', scheduler_counter)
            for counter, (img_inputs, meta_inputs, labels_slide_id) in enumerate(dataloaders[phase]): 
                if test_flag and counter >5: 
                    break
               
                labels = labels_slide_id[:,0] # (label, slide_id, col, row)
                slide_ids = labels_slide_id[:,1]
                cols = labels_slide_id[:,2]
                rows = labels_slide_id[:,3]
                
                img_inputs = img_inputs.to(device)
                meta_inputs = meta_inputs.to(device)
                labels = labels.to(device)
             
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): 
                    if mode == 'base': 
                        outputs = model(img_inputs) # Output is the result of fully connected layer (two neurons), no softmax
                    else:
                        # Inportance, or Concatenation  
                        #print(meta_inputs.shape)
                        outputs = model(img_inputs, meta_inputs) # output is a value between 0 and 1, describing the probability of melanoma
                  
                    if outputs.dim() == 1: # BCELoss; bei approach 1 kommt hier ein Fehler ... 
                        preds = 1 * (outputs >= 0.5) 
                        labels = labels.float()
                        #print(labels)
                    else:
                        # if Cross Entropy 
                        outputs = torch.softmax(outputs,dim=1) # turning the output into 'probabilities'
                        _, preds = torch.max(outputs, 1)
                    
                    y_preds.append(preds.detach().cpu().numpy())
                    y_trues.append(labels.detach().cpu().numpy())
                    slide_id_list.append(slide_ids.detach().cpu().numpy())
                    y_scores.append(outputs.detach().cpu().numpy())
                    
                    loss = criterion(outputs, labels)
                 
                    if phase == 'train': 
                        loss.backward()
                        optimizer.step()
                        lr.append(scheduler.get_last_lr())
                        scheduler_counter +=1 
                        scheduler.step() # scheduler step after each minibatch! 
                       
                    running_loss += loss.item()*img_inputs.size(0)
                    
            
            # Balanced accuracy on tile level 
            y_preds = np.concatenate(y_preds)
            y_trues = np.concatenate(y_trues)
            slide_id_list = np.concatenate(slide_id_list)
            y_scores = np.concatenate(y_scores)
            bal_acc_tiles = balanced_accuracy_score(y_trues, y_preds)
            epoch_loss = running_loss/ len(image_datasets[phase])
            bal_acc_slides = calculate_slide_acc(y_scores, slide_id_list, y_trues)
            #bal_acc_slides = calculate_slide_acc(y_preds, slide_id_list, y_trues)
            print('Phase {}, epoch loss {:.5f}, bal_acc_tiles {:.5f}, bal_acc_slides {:.5f}'.format(phase, epoch_loss, bal_acc_tiles, bal_acc_slides))
            
            # Saving epoch results 
            if phase == 'train': 
                trainings_loss.append(epoch_loss)
                balanced_accuracy_tiles_train.append(bal_acc_tiles)
                balanced_accuracy_slides_train.append(bal_acc_slides)
            else: 
                validation_loss.append(epoch_loss)
                balanced_accuracy_tiles_val.append(bal_acc_tiles)
                balanced_accuracy_slides_val.append(bal_acc_slides)
            
            #deep copy model
            #if phase == 'val' and epoch_loss < epoch_lowest_loss: # Auf loss umsteigen?
            if phase == 'val' and bal_acc_tiles > best_bal_acc and epoch > 5: 
                best_bal_acc = bal_acc_tiles
                epoch_best_acc = epoch 
                best_model_wts = copy.deepcopy(model.state_dict())
                best_results['fold_'+str(fold)] = {'epoch': epoch,  'bal_acc_tiles': bal_acc_tiles, 'bal_acc_slides': bal_acc_slides}
             
                save_model_path = result_path + 'models/'
                if not os.path.exists(save_model_path): 
                    os.makedirs(save_model_path)
                #date = str(datetime.now().strftime(\"%d%m%Y\"))
                model_name = 'best_model_fold_'+str(fold)
                torch.save(model.state_dict(), save_model_path+model_name)
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    #print('Best BalAcc on Validation: {:.5f} achieved in epoch {}'.format(best_bal_acc, epoch_best))
    print('Best BalAcc on Validation: {:.5f} achieved in epoch {}'.format(best_bal_acc, epoch_best_acc))
    metric_dict['fold_'+str(fold)] = {'loss_train': trainings_loss, 
                                      'bal_acc_tiles_train': balanced_accuracy_tiles_train, 
                                      'bal_acc_slides_train': balanced_accuracy_slides_train,
                                      'loss_val': validation_loss, 
                                      'bal_acc_tiles_val': balanced_accuracy_tiles_val, 
                                      'bal_acc_slides_val': balanced_accuracy_slides_val, 
                                      'lr': lr}
    print('Loading the models best parameters')
    model.load_state_dict(best_model_wts)
    
    return model, best_results, metric_dict

def test_loop_v1(model, dataloaders, image_datasets, test_results, device, fold, mode, phase='test'):
    
    model.eval() 
    
    running_loss = 0.
    y_preds = []
    y_trues = []
    slide_id_list = []
    y_scores = []

    for counter, (img_inputs, meta_inputs, labels_slide_id) in enumerate(dataloaders[phase]): 
        labels = labels_slide_id[:,0] 
        slide_ids = labels_slide_id[:,1]
        cols = labels_slide_id[:,2]
        rows = labels_slide_id[:,3]
                
        img_inputs = img_inputs.to(device)
        meta_inputs = meta_inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(phase == 'train'): 
            if mode == 'base': 
                outputs = model(img_inputs) 
            else:
                # Inportance, or Concatenation, multiplication  
                outputs = model(img_inputs, meta_inputs) 
                
            if outputs.dim() == 1: 
                # BCELoss
                preds = 1 * (outputs >= 0.5) 
                labels = labels.float()
                
            else:
                # Cross Entropy
                outputs = torch.softmax(outputs,dim=1) 
                _, preds = torch.max(outputs, 1)
            
            y_preds.append(preds.detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
            slide_id_list.append(slide_ids.detach().cpu().numpy())
            y_scores.append(outputs.detach().cpu().numpy())          
   
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    slide_id_list = np.concatenate(slide_id_list)
    y_scores = np.concatenate(y_scores)
    # tile level 
    bal_acc_tiles = balanced_accuracy_score(y_trues, y_preds)
  
    # slide level 
    bal_acc_slides = calculate_slide_acc(y_scores, slide_id_list, y_trues)
    print('Phase {}, bal_acc_tiles {:.5f}, bal_acc_slides {:.5f}'.format(phase, bal_acc_tiles, bal_acc_slides))
    test_results['fold_'+str(fold)] = {'bal_acc_t': bal_acc_tiles, 'bal_acc_s': bal_acc_slides}
    
    return test_results 

def test_loop_v2(model, dataloaders, criterion, image_datasets, test_results, device, fold, mode, phase='test'):
    
    model.eval() 
    
    running_loss = 0.
    y_preds = []
    y_trues = []
    slide_id_list = []
    y_scores = []

    
    for img_inputs, meta_inputs, labels_slide_id in tqdm(dataloaders[phase]): 
        labels = labels_slide_id[:,0] # (label, slide_id, col, row)
        slide_ids = labels_slide_id[:,1]
        cols = labels_slide_id[:,2]
        rows = labels_slide_id[:,3]
                
        img_inputs = img_inputs.to(device)
        meta_inputs = meta_inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(phase == 'train'): 
            if mode == 'base': 
                outputs = model(img_inputs) 
            else:
                # Inportance, or Concatenation, multiplication  
                outputs = model(img_inputs, meta_inputs) 
                
            if outputs.dim() == 1: # BCELoss
                preds = 1 * (outputs >= 0.5) 
                labels = labels.float()
            else:
                # if Cross Entropy 
                outputs = torch.softmax(outputs,dim=1) 
                _, preds = torch.max(outputs, 1)
            
            y_preds.append(preds.detach().cpu().numpy())
            y_trues.append(labels.detach().cpu().numpy())
            slide_id_list.append(slide_ids.detach().cpu().numpy())
            y_scores.append(outputs.detach().cpu().numpy())          
   
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    slide_id_list = np.concatenate(slide_id_list)
    y_scores = np.concatenate(y_scores)
    # tile level 
    bal_acc_tiles = balanced_accuracy_score(y_trues, y_preds)
    # slide level 
    bal_acc_slides, slides_included, slides_label, slides_preds, slides_output  = calculate_slide_acc_v1(y_scores, slide_id_list, y_trues)
    print('Phase {}, bal_acc_tiles {:.5f}, bal_acc_slides {:.5f}'.format(phase, bal_acc_tiles, bal_acc_slides))
    test_results['fold_'+str(fold)] = {'bal_acc_t': bal_acc_tiles, 'bal_acc_s': bal_acc_slides}
    
    return test_results, y_scores, slides_label, slides_preds, slides_output, slides_included

def calculate_slide_acc(y_scores, slide_id_list, y_trues): 
    slides_included = np.unique(slide_id_list)
    slides_label = np.zeros(int(len(slides_included)))
    if y_scores.ndim == 1: 
        slides_output = np.zeros(len(slides_included))
        for counter, slide in enumerate(slides_included): 
            idx = np.where(slide_id_list == slide)[0]  
            tiles_probs = y_scores[idx].sum(axis=0)/len(idx) 
            slides_output[counter] = tiles_probs  
            slides_label[counter] = y_trues[idx[0]]
        slides_preds = 1 * (slides_output >= 0.5)
       
    else: # base and approach 1 
        slides_output = np.zeros((len(slides_included),2))
        for counter, slide in enumerate(slides_included): 
            idx = np.where(slide_id_list == slide)[0] 
            tiles_probs = y_scores[idx,:].sum(axis=0)/len(idx) 
            slides_output[counter] = tiles_probs 
            slides_label[counter] = y_trues[idx[0]]
        slides_preds = np.argmax(slides_output, 1)  
     
    bal_acc_slides = balanced_accuracy_score(slides_label, slides_preds)
    return bal_acc_slides

def calculate_slide_acc_v1(y_scores, slide_id_list, y_trues): 
    slides_included = np.unique(slide_id_list)
    slides_label = np.zeros(int(len(slides_included)))
    
    if y_scores.ndim == 1: # if BCE (approach 2, appraoch 3)
        slides_output = np.zeros(len(slides_included))
        for counter, slide in enumerate(slides_included): 
            idx = np.where(slide_id_list == slide)[0] 
            tiles_probs = y_scores[idx].sum(axis=0)/len(idx) 
            slides_output[counter] = tiles_probs 
            slides_label[counter] = y_trues[idx[0]]
        slides_preds = 1 * (slides_output >= 0.5)
       
    else: # base and approach 1 
        slides_output = np.zeros((len(slides_included),2))
        for counter, slide in enumerate(slides_included): 
            idx = np.where(slide_id_list == slide)[0] 
            tiles_probs = y_scores[idx,:].sum(axis=0)/len(idx) 
            slides_output[counter] = tiles_probs 
            slides_label[counter] = y_trues[idx[0]]
        slides_preds = np.argmax(slides_output, 1)  
     
    bal_acc_slides = balanced_accuracy_score(slides_label, slides_preds)
    
    return bal_acc_slides, slides_included, slides_label, slides_preds, slides_output 
    
