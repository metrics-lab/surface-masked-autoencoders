# -*- coding: utf-8 -*-
# @Author: Simon Dahan
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import os
import argparse
import yaml
import sys
import timm
from datetime import datetime
#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.sit import SiT
from models.ms_sit import MSSiT
from models.ms_sit_shifted import MSSiT_shifted
#from models.sphericalunet import sphericalunet_regression

from tools.utils import load_weights_imagenet, logging_sit, logging_ms_sit, logging_spherical_unet, plot_regression_results_UKB, plot_regression_results_dHCP, plot_regression_results_HCP
from tools.utils import get_data_path, get_dataloaders, get_dimensions, get_scheduler
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, r2_score
from scipy.stats import pearsonr


def train(config):

    #############################
    ######     CONFIG      ######
    #############################

    print('')
    print('#'*30)
    print('########### Config ###########')
    print('#'*30)
    print('')

    #mesh_resolution
    ico_mesh = config['mesh_resolution']['ico_mesh']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    #data
    dataset = config['data']['dataset']
    task = config['data']['task']
    configuration = config['data']['configuration']
    hemi = config['data']['hemi']

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    loss = config['training']['loss']
    epochs = config['training']['epochs']
    val_epoch = config['training']['val_epoch']
    testing = config['training']['testing']
    testing_debug = config['training']['testing_debug']
    log_training_epoch = config['training']['log_training_epoch']
    use_confounds = config['training']['use_confounds']
    early_stopping = config['training']['early_stopping']

    if hemi == 'full':  ### TO IMPLEMENT
        num_patches*=2

    if task == 'sex' or task == 'sex_msmall':
        classification_task = True
    else: 
        classification_task = False

    try:
        data_path = get_data_path(config)
    except:
        raise("can't get data path")

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print('gpu: {}'.format(device))   
    print('dataset: {}'.format(dataset))  
    print('use confounds: {}'.format(use_confounds))
    print('task: {}'.format(task))  
    print('model: {}'.format(config['MODEL']))
    print('configuration: {}'.format(configuration))  
    print('data path: {}'.format(data_path))
    print('classification task: {}'.format(classification_task))

    ##############################
    ######     DATASET      ######
    ##############################

    if config['MODEL'] == 'sit' or config['MODEL'] == 'ms-sit':
        print('')
        print('Mesh resolution - ico {}'.format(ico_mesh))
        print('Grid resolution - ico {}'.format(ico_grid))
        print('Number of patches - {}'.format(num_patches))
        print('Number of vertices - {}'.format(num_vertices))
        print('Reorder patches: {}'.format(config['mesh_resolution']['reorder']))
        print('')

    try:
        train_loader, val_loader, test_loader = get_dataloaders(config,data_path)
        print('')
    except:
        raise("can't get dataloaders")

    ##############################
    ######      LOGGING     ######
    ##############################

    print('')
    print('#'*30)
    print('########## Logging ###########')
    print('#'*30)
    print('')

    # creating folders for logging. 
    if config['MODEL'] == 'sit':
        folder_to_save_model = logging_sit(config)
    elif config['MODEL'] == 'ms-sit':
        folder_to_save_model = logging_ms_sit(config) 
    elif config['MODEL'] == 'spherical-unet':
        folder_to_save_model = logging_spherical_unet(config)
    else:
        raise('not implemented yet')
    
    try:
        os.makedirs(folder_to_save_model,exist_ok=False)
        print('Creating folder: {}'.format(folder_to_save_model))
    except OSError:
        print('folder already exist: {}'.format(folder_to_save_model))
    
    #tensorboard
    writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################

    print('')
    print('#'*30)
    print('######### Init model #########')
    print('#'*30)
    print('')

    if config['MODEL'] == 'sit' or config['MODEL'] == 'ms-sit':    

        T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config)

    if config['MODEL'] == 'sit':
        model = SiT(dim=config['transformer']['dim'],
                        depth=config['transformer']['depth'],
                        heads=config['transformer']['heads'],
                        pool=config['transformer']['pool'], 
                        num_patches=N,
                        num_classes=config['transformer']['num_classes'],
                        num_channels=T,
                        num_vertices=V,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'],
                        use_pe=config['transformer']['use_pos_embedding'],
                        bottleneck_dropout=bottleneck_dropout,
                        use_bottleneck=use_bottleneck,
                        use_confounds=use_confounds,
                        weights_init=config['transformer']['init_weights'],
                        use_class_token=config['transformer']['use_class_token'],
                        trainable_pos_emb=config['transformer']['trainable_pos_emb'],
                        no_class_emb = config['transformer']['no_class_emb'],)
    
    elif config['MODEL'] == 'ms-sit':
        if config['transformer']['shifted_attention']:
            print('*** using shifted attention with shifting factor {} ***'.format(config['transformer']['window_size_factor']))
            model = MSSiT_shifted(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                                    num_channels=T,
                                    num_classes=config['transformer']['num_classes'],
                                    embed_dim=config['transformer']['dim'],
                                    depths=config['transformer']['depth'],
                                    num_heads=config['transformer']['heads'],
                                    window_size=config['transformer']['window_size'],
                                    window_size_factor=config['transformer']['window_size_factor'],
                                    mlp_ratio=config['transformer']['mlp_ratio'],
                                    qkv_bias=True,
                                    qk_scale=True,
                                    dropout=config['transformer']['dropout'],
                                    attention_dropout=config['transformer']['attention_dropout'],
                                    dropout_path=config['transformer']['dropout_path'],
                                    norm_layer=nn.LayerNorm,
                                    use_pos_emb=config['transformer']['use_pos_emb'],
                                    patch_norm=True,
                                    use_confounds=use_confounds,
                                    device=device
                                    )
        
        else: 
            model = MSSiT(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                            num_channels=T,
                            num_classes=config['transformer']['num_classes'],
                            embed_dim=config['transformer']['dim'],
                            depths=config['transformer']['depth'],
                            num_heads=config['transformer']['heads'],
                            window_size=config['transformer']['window_size'],
                            mlp_ratio=config['transformer']['mlp_ratio'],
                            qkv_bias=True,
                            qk_scale=True,
                            dropout=config['transformer']['dropout'],
                            attention_dropout=config['transformer']['attention_dropout'],
                            dropout_path=config['transformer']['dropout_path'],
                            norm_layer=nn.LayerNorm,
                            use_pos_emb=config['transformer']['use_pos_emb'],
                            patch_norm=True,
                            use_confounds=use_confounds,
                            device=device
                            )

    elif config['MODEL'] == 'spherical-unet':
        model = sphericalunet_regression(num_features = config['spherical-unet']['num_features'],
                                         in_channels=len(config['spherical-unet']['channels']))
        
    print('')

    if config['training']['init_weights']=='ssl_mae':
        print('Loading weights from self-supervision training MAE from: {}'.format(config['weights']['ssl_mae']))
        model.load_state_dict(torch.load(config['weights']['ssl_mae'],map_location=device)['model_state_dict'],strict=True)
    elif config['training']['init_weights']=='ssl_smae':
        print('Loading weights from self-supervision training MAE from: {}'.format(config['weights']['ssl_smae']))
        strict = False if task=='birth_age' else True
        model.load_state_dict(torch.load(config['weights']['ssl_smae'],map_location=device)['model_state_dict'],strict=strict)
    elif config['training']['init_weights']=='ssl_mpp':
        print('Loading weights from self-supervision training MPP from: {}'.format(config['weights']['ssl_mpp']))
        #import pdb;pdb.set_trace()
        #model.load_state_dict(torch.load(config['weights']['ssl_mpp'],map_location=device)['model_state_dict'],strict=False)
        strict = False if task=='birth_age' else True
        model.load_state_dict(torch.load(config['weights']['ssl_mpp'],map_location=device)['model_state_dict'],strict=strict)
        #import pdb;pdb.set_trace()

    elif config['training']['init_weights']=='imagenet':
        print('Loading weights from imagenet pretraining')
        model_trained = timm.create_model(config['weights']['imagenet'], pretrained=True)
        new_state_dict = load_weights_imagenet(model.state_dict(),model_trained.state_dict(),config['transformer']['depth'])
        model.load_state_dict(new_state_dict)
    
    else:
        print('Training from scratch')
        
    if config['training']['finetuning']==False:
            print('freezing all layers except mlp head')
            #import pdb;pdb.set_trace()
            for j, (name, param) in enumerate(model.named_parameters()):
                if 'mlp_head' not in name:
                    param.requires_grad = False
                else:
                    print(name)
                    param.requires_grad = True
    #import pdb;pdb.set_trace()


    model.to(device)
  
    print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('')

    #####################################
    #######     OPTIMISATION      #######
    #####################################

    if config['optimisation']['optimiser']=='SGD':
        print('Using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
                                                
    elif config['optimisation']['optimiser']=='Adam':
        print('Using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR,
                                weight_decay=config['Adam']['weight_decay'])
      
    elif config['optimisation']['optimiser']=='AdamW':
        print('Using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')

    ##############################
    #######     LOSS       #######
    ##############################

    if classification_task:  
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        if loss == 'mse':
            criterion = nn.MSELoss(reduction='mean')
        elif loss == 'l1':
            criterion = nn.L1Loss(reduction='mean')
        
    print('Using {} criterion'.format(criterion))
    
    ###################################
    #######     SCHEDULING      #######
    ###################################
    it_per_epoch = np.ceil(len(train_loader))

    scheduler = get_scheduler(config,it_per_epoch,  optimizer)

    ##############################
    ######     TRAINING     ######
    ##############################

    print('')
    print('#'*30)
    print('###### Starting training #####')
    print('#'*30)
    print('')

    best_accuracy = 0  # classification task
    best_mae = 1000000000000000 # regression task
    best_loss = 1000000000000000 # classification and regression tasks
    c_early_stop = 0

    for epoch in range(epochs):

        running_loss = 0

        model.train()

        targets_ =  []
        preds_ = []

        for i, data in enumerate(train_loader):
 
            if use_confounds:
                inputs, labels = data[0].to(device), data[1].to(device)
                if config['MODEL']=='spherical-unet':
                    inputs = inputs.permute(2,1,0)
                targets, confounds =  labels[:,0], labels[:,1]
                #outputs, att  = model(inputs, confounds)
                outputs  = model(inputs, confounds)
            
            else:
                inputs, targets = data[0].to(device), data[1].to(device)
                if config['MODEL']=='spherical-unet':
                    inputs = inputs.permute(2,1,0)
                #outputs, att = model(inputs)
                outputs = model(inputs)
            
            optimizer.zero_grad()
            
            loss = criterion(outputs.squeeze(), targets)

            writer.add_scalar('loss/train_it', loss.item(), epoch*it_per_epoch + i +1)

            if config['optimisation']['use_scheduler']:
                if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                    if config['optimisation']['warmup']:
                        if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                            writer.add_scalar('LR',scheduler.get_lr()[0], epoch*it_per_epoch + i +1 )
                        else:
                            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                    else:
                        writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                else:
                    scheduler.step()
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
            else:
                if config['optimisation']['warmup']:
                    scheduler.step()
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], epoch*it_per_epoch + i +1 )

            loss.backward()
            optimizer.step()

            running_loss += loss.item() #average over batch (default)

            if classification_task: 
                      
                targets_ += list(np.array(targets.cpu().numpy(),dtype=np.int32))
                preds_ += list(np.array(np.round(torch.sigmoid(outputs).reshape(-1).cpu().detach().numpy()),dtype=np.int32))

            else: # regression task
                
                targets_.append(targets.cpu().numpy())
                preds_.append(outputs.reshape(-1).cpu().detach().numpy())
            
        loss_train_epoch = running_loss / (i+1)

        writer.add_scalar('loss/train', loss_train_epoch, epoch+1)

        if classification_task:  

            accuracy_epoch = accuracy_score(preds_,targets_)
            balanced_accuracy_epoch = balanced_accuracy_score(preds_,targets_)

            writer.add_scalar('accuracy/train',accuracy_epoch, epoch+1)
            writer.add_scalar('balanced_accuracy/train',balanced_accuracy_epoch, epoch+1)

            if (epoch+1)%log_training_epoch==0:
                if config['optimisation']['use_scheduler']:
                    if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                        if config['optimisation']['warmup']:
                            if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                                print('| Epoch - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, round(accuracy_epoch,4), scheduler.get_lr()[0] ))
                            else:
                                print('| Epoch - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, round(accuracy_epoch,4),optimizer.param_groups[0]['lr'] ))
                        else:
                            print('| Epoch - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, round(accuracy_epoch,4),optimizer.param_groups[0]['lr'] ))
                    else:
                        print('| Epoch - {} | Loss - {:.4f} | accuracy - {:.2f} | balanced accuracy - {:.2f}% | LR - {}'.format(epoch+1, loss_train_epoch, accuracy_epoch*100, balanced_accuracy_epoch*100,scheduler.get_last_lr()[0] ))
                else:
                    print('| Epoch - {} | Loss - {:.4f} | accuracy - {:.2f}% | balanced accuracy - {:.2f}% | LR - {}'.format(epoch+1, loss_train_epoch, accuracy_epoch*100, balanced_accuracy_epoch*100 ,optimizer.param_groups[0]['lr']))

        else:
                
            mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))
            r2_epoch = r2_score(np.concatenate(targets_) , np.concatenate(preds_))
            correlation = pearsonr(np.concatenate(targets_).reshape(-1),np.concatenate(preds_).reshape(-1))[0]

            writer.add_scalar('mae/train',mae_epoch, epoch+1)
            writer.add_scalar('r2/train',r2_epoch, epoch+1)
            writer.add_scalar('correlation/train',correlation, epoch+1)

            if (epoch+1)%log_training_epoch==0:
                if dataset == 'UKB':
                    plot_regression_results_UKB(np.concatenate(preds_), np.concatenate(targets_), os.path.join(folder_to_save_model,'results_train'), epoch+1)
                elif dataset == 'HCP':
                    plot_regression_results_HCP(np.concatenate(preds_), np.concatenate(targets_), os.path.join(folder_to_save_model,'results_train'), epoch+1)

                if config['optimisation']['use_scheduler']:
                    if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                        if config['optimisation']['warmup']:
                            if (epoch+1)<config['optimisation']['nbr_step_warmup']:
                                print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(epoch+1,loss_train_epoch, round(mae_epoch,4), scheduler.get_lr()[0] ))
                            else:
                                print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
                        else:
                            print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
                    else:
                        print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | R2 - {:.4f} | Corr - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, mae_epoch, r2_epoch, correlation,  scheduler.get_last_lr()[0] ))
                else:
                    print('| Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | R2 - {:.4f} | Corr - {:.4f} | LR - {}'.format(epoch+1, loss_train_epoch, mae_epoch, r2_epoch, correlation,  optimizer.param_groups[0]['lr']))

        ##############################
        ######    VALIDATION    ######
        ##############################

        if (epoch+1)%val_epoch==0:

            running_val_loss = 0
            model.eval()

            with torch.no_grad():

                targets_ = []
                preds_ = []

                for i, data in enumerate(val_loader):
                    if use_confounds:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        if config['MODEL']=='spherical-unet':
                            inputs = inputs.permute(2,1,0)
                        targets, confounds =  labels[:,0], labels[:,1]
                        outputs = model(inputs, confounds)
                    else:
                        inputs, targets = data[0].to(device), data[1].to(device)
                        if config['MODEL']=='spherical-unet':
                            inputs = inputs.permute(2,1,0)
                        outputs = model(inputs)

                    loss = criterion(outputs[0], targets)
                    running_val_loss += loss.item()

                    if classification_task:  
                        targets_ += list(np.array(targets.cpu().numpy(),dtype=np.int32))
                        preds_ += list(np.array(np.round(torch.sigmoid(outputs).reshape(-1).cpu().detach().numpy()),dtype=np.int32))

                    else:
                        targets_.append(targets.cpu().numpy())
                        preds_.append(outputs.reshape(-1).cpu().numpy())


            loss_val_epoch = running_val_loss/(i+1)

            writer.add_scalar('loss/val',loss_val_epoch , epoch+1)

            if classification_task:  

                accuracy_val_epoch =accuracy_score(targets_,preds_)
                balanced_accuracy_val_epoch = balanced_accuracy_score(targets_,preds_)

                writer.add_scalar('accuracy/val',accuracy_val_epoch, epoch+1)
                writer.add_scalar('balanced_accuracy/val',balanced_accuracy_val_epoch, epoch+1)
                
                print('| Validation | Epoch - {} | Loss - {:.4f} | accuracy - {:.4f} | balanced accuracy - {:.4f} |'.format(epoch+1, loss_val_epoch, accuracy_val_epoch, balanced_accuracy_val_epoch))
                
                cm = confusion_matrix(targets_,preds_ )
                print(cm)

                if loss_val_epoch<best_loss:

                    best_accuracy = accuracy_val_epoch
                    best_loss = loss_val_epoch
                    best_epoch = epoch+1
                    c_early_stop = 0

                    config['logging']['folder_model_saved'] = folder_to_save_model
                    config['results'] = {}
                    config['results']['best_val_accuracy'] = float(best_accuracy)
                    config['results']['best_val_loss'] = float(best_loss)
                    config['results']['best_val_balanced_accuracy'] = float(balanced_accuracy_val_epoch)
                    config['results']['best_val_epoch'] = best_epoch
                    config['results']['training_finished'] = False

                    ###############################
                    ######    SAVING CKPT    ######
                    ###############################

                    if config['training']['save_ckpt']:
                        torch.save({'epoch': epoch,
                                    'model_state_dict':model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss_train_epoch,
                                    },
                                    os.path.join(folder_to_save_model,'checkpoint_best.pth'))
                elif early_stopping:
                        c_early_stop += 1

            else:

                mae_val_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))
                r2_val_epoch = r2_score(np.concatenate(targets_) , np.concatenate(preds_) )
                correlation_val = pearsonr(np.concatenate(targets_).reshape(-1),np.concatenate(preds_).reshape(-1))[0]

                writer.add_scalar('mae/val',mae_val_epoch, epoch+1)
                writer.add_scalar('r2/val',r2_val_epoch, epoch+1)
                writer.add_scalar('correlation/val',correlation_val, epoch+1)

                if dataset == 'UKB':
                    plot_regression_results_UKB(np.concatenate(preds_), np.concatenate(targets_), os.path.join(folder_to_save_model,'results_val'), epoch+1)

                elif dataset == 'dHCP':
                    plot_regression_results_dHCP(np.concatenate(preds_), np.concatenate(targets_), os.path.join(folder_to_save_model,'results_val'), epoch+1)
                
                elif dataset == 'HCP':
                    plot_regression_results_HCP(np.concatenate(preds_), np.concatenate(targets_), os.path.join(folder_to_save_model,'results_val'), epoch+1)
                    
                print('| Validation | Epoch - {} | Loss - {:.4f} | MAE - {:.4f} | Corr - {:.4f} | R2 - {:.4f} |'.format(epoch+1, loss_val_epoch, mae_val_epoch, correlation_val, r2_val_epoch))

                if mae_val_epoch < best_mae or loss_val_epoch < best_loss:

                    best_mae = mae_val_epoch
                    best_loss = loss_val_epoch
                    best_epoch = epoch+1
                    c_early_stop = 0

                    config['logging']['folder_model_saved'] = folder_to_save_model
                    config['results'] = {}
                    config['results']['best_val_mae'] = float(best_mae)
                    config['results']['best_val_loss'] = float(best_loss)
                    config['results']['best_val_r2'] = float(r2_val_epoch)
                    config['results']['best_val_epoch'] = best_epoch
                    config['results']['training_finished'] = False 

                    if testing_debug:

                        model.eval()

                        print('starting testing')

                        with torch.no_grad():

                            targets_ = []
                            preds_ = []

                            for i, data in enumerate(test_loader):
                                if use_confounds:
                                    inputs, labels = data[0].to(device), data[1].to(device)
                                    if config['MODEL']=='spherical-unet':
                                        inputs = inputs.permute(2,1,0)
                                    targets, confounds =  labels[:,0], labels[:,1]
                                    outputs = model(inputs, confounds)
                                else:
                                    inputs, targets = data[0].to(device), data[1].to(device)
                                    if config['MODEL']=='spherical-unet':
                                        inputs = inputs.permute(2,1,0)
                                    outputs = model(inputs)

                                targets_.append(targets.cpu().numpy())
                                preds_.append(outputs.reshape(-1).cpu().numpy())

                            mae_test_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))
                            r2_test_epoch = r2_score(np.concatenate(targets_) , np.concatenate(preds_) )
                            correlation_test = pearsonr(np.concatenate(targets_).reshape(-1),np.concatenate(preds_).reshape(-1))[0]

                            print('| Testing debug | MAE - {:.4f} | Corr - {:.4f} | R2 - {:.4f}'.format( mae_test_epoch,correlation_test, r2_test_epoch))


                    ###############################
                    ######    SAVING CKPT    ######
                    ###############################

                    if config['training']['save_ckpt']:
                        torch.save({'epoch': epoch,
                                    'model_state_dict':model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss_train_epoch,
                                    },
                                    os.path.join(folder_to_save_model,'checkpoint_best.pth'))
                elif early_stopping:
                        c_early_stop += 1

            with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                yaml.dump(config, yaml_file)

        if early_stopping and (c_early_stop>=early_stopping):
            print('stop training - early stopping')
            break

        ##################################
        ######   UPDATE SCHEDULER  #######
        ##################################

        if config['optimisation']['use_scheduler']:
            if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=mae_val_epoch)
              
    
    if classification_task:  
        print('Final results: best model obtained at epoch {} - best accuracy validation {}'.format(best_epoch,best_accuracy))
    else:
        print('Final results: best model obtained at epoch {} - best MAE validation {}'.format(best_epoch,best_mae))

    if early_stopping and (c_early_stop>=early_stopping):
        config['results']['training_finished'] = 'early stopping' 
    else:
        config['results']['training_finished'] = True 

    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################

    if config['training']['save_ckpt']:
        torch.save({'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train_epoch,
                    },
                    os.path.join(folder_to_save_model,'checkpoint_final.pth'))
    
    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)

    ##############################
    ######     TESTING      ######
    ##############################

    if testing:
        
        print('')
        print('#'*30)
        print('###### Starting testing #####')
        print('#'*30)
        print('')

        del train_loader
        del val_loader
        del model
        #torch.cuda.empty_cache()

        if config['MODEL'] == 'sit' or config['MODEL'] == 'ms-sit':    

            T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config)

        if config['MODEL'] == 'sit':
            test_model = SiT(dim=config['transformer']['dim'],
                            depth=config['transformer']['depth'],
                            heads=config['transformer']['heads'],
                            pool=config['transformer']['pool'], 
                            num_patches=N,
                            num_classes=config['transformer']['num_classes'],
                            num_channels=T,
                            num_vertices=V,
                            dim_head=config['transformer']['dim_head'],
                            dropout=config['transformer']['dropout'],
                            emb_dropout=config['transformer']['emb_dropout'],
                            use_pe=config['transformer']['use_pos_embedding'],
                            bottleneck_dropout=bottleneck_dropout,
                            use_bottleneck=use_bottleneck,
                            use_confounds=use_confounds,
                            weights_init=config['transformer']['init_weights'],
                            use_class_token=config['transformer']['use_class_token'],
                            trainable_pos_emb=config['transformer']['trainable_pos_emb'],
                            no_class_emb = config['transformer']['no_class_emb'],)
        
        elif config['MODEL'] == 'ms-sit':
            if config['transformer']['shifted_attention']:
                print('*** using shifted attention with shifting factor {}***'.format(config['transformer']['window_size_factor']))
                test_model = MSSiT_shifted(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                                        num_channels=T,
                                        num_classes=config['transformer']['num_classes'],
                                        embed_dim=config['transformer']['dim'],
                                        depths=config['transformer']['depth'],
                                        num_heads=config['transformer']['heads'],
                                        window_size=config['transformer']['window_size'],
                                        window_size_factor=config['transformer']['window_size_factor'],
                                        mlp_ratio=config['transformer']['mlp_ratio'],
                                        qkv_bias=True,
                                        qk_scale=True,
                                        dropout=config['transformer']['dropout'],
                                        attention_dropout=config['transformer']['attention_dropout'],
                                        dropout_path=config['transformer']['dropout_path'],
                                        norm_layer=nn.LayerNorm,
                                        use_pos_emb=config['transformer']['use_pos_emb'],
                                        patch_norm=True,
                                        use_confounds=use_confounds,
                                        device=device
                                        )
            
            else: 
                test_model = MSSiT(ico_init_resolution=config['mesh_resolution']['ico_grid'],
                                num_channels=T,
                                num_classes=config['transformer']['num_classes'],
                                embed_dim=config['transformer']['dim'],
                                depths=config['transformer']['depth'],
                                num_heads=config['transformer']['heads'],
                                window_size=config['transformer']['window_size'],
                                mlp_ratio=config['transformer']['mlp_ratio'],
                                qkv_bias=True,
                                qk_scale=True,
                                dropout=config['transformer']['dropout'],
                                attention_dropout=config['transformer']['attention_dropout'],
                                dropout_path=config['transformer']['dropout_path'],
                                norm_layer=nn.LayerNorm,
                                use_pos_emb=config['transformer']['use_pos_emb'],
                                patch_norm=True,
                                use_confounds=use_confounds,
                                device=device
                                )

        elif config['MODEL'] == 'spherical-unet':
            test_model = sphericalunet_regression(num_features = config['spherical-unet']['num_features'],
                                            in_channels=len(config['spherical-unet']['channels']))

        test_model.load_state_dict(torch.load(os.path.join(folder_to_save_model,'checkpoint_best.pth'))['model_state_dict'])
        print('')
        print('Successfully loading model')
        print('')
        test_model.to(device)

        test_model.eval()

        with torch.no_grad():

            targets_ = []
            preds_ = []

            for i, data in enumerate(test_loader):

                if use_confounds:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    if config['MODEL']=='spherical-unet':
                            inputs = inputs.permute(2,1,0)
                    targets, confounds =  labels[:,0], labels[:,1]
                    outputs = test_model(inputs, confounds)
                else:
                    inputs, targets = data[0].to(device), data[1].to(device)
                    if config['MODEL']=='spherical-unet':
                        inputs = inputs.permute(2,1,0)
                    outputs = test_model(inputs)

                if classification_task:  
                    targets_ += list(np.array(targets.cpu().numpy(),dtype=np.int32))
                    preds_ += list(np.array(np.round(torch.sigmoid(outputs).reshape(-1).cpu().detach().numpy()),dtype=np.int32))
                else:
                    targets_.append(targets.cpu().numpy())
                    preds_.append(outputs.reshape(-1).cpu().numpy())                    
                    
        if classification_task:  
            accuracy_test =accuracy_score(targets_,preds_)
            balanced_accuracy_test = balanced_accuracy_score(targets_,preds_)
            print('| Testing | accuracy - {:.4f} | balanced accuracy - {:.4f} |'.format(accuracy_test,balanced_accuracy_test))
            config['results']['accuracy_testing'] = float(accuracy_test)
            config['results']['balanced_accuracy_testing'] = float(balanced_accuracy_test)

        else:
            mae_test_epoch = np.mean(np.abs(np.concatenate(targets_)- np.concatenate(preds_)))
            r2_test_epoch = r2_score(np.concatenate(targets_) , np.concatenate(preds_) )
            correlation_test = pearsonr(np.concatenate(targets_).reshape(-1),np.concatenate(preds_).reshape(-1))[0]

            print('| Testing | MAE - {:.4f} | Corr - {:.4f} | R2 - {:.4f}'.format( mae_test_epoch,correlation_test, r2_test_epoch))
            config['results']['mae_testing'] = float(mae_test_epoch)
            config['results']['r2_testing'] = float(r2_test_epoch)
            config['results']['correlation_test'] = float(correlation_test)

            if dataset == 'UKB':
                plot_regression_results_UKB(preds_, targets_, os.path.join(folder_to_save_model,'results_test'), best_epoch)
            elif dataset == 'dHCP':
                plot_regression_results_dHCP(preds_, targets_, os.path.join(folder_to_save_model,'results_test'), best_epoch)
            elif dataset == 'HCP':
                plot_regression_results_HCP(preds_, targets_, os.path.join(folder_to_save_model,'results_test'), best_epoch)

        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)

        df = pd.DataFrame()
        df['preds'] = np.concatenate(preds_).reshape(-1)
        df['targets'] = np.concatenate(targets_).reshape(-1)
        df.to_csv(os.path.join(folder_to_save_model, 'preds_test.csv'))

    else:
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call training
    train(config)
