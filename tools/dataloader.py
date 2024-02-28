import os
import sys

import numpy as np
import nibabel as nb
import pandas as pd

sys.path.append('./')
sys.path.append('./tools')

from tools.datasets import *
from tools.samplers import new_sampler_HCP_fluid_intelligence,sampler_preterm_birth_age, sampler_preterm_scan_age, sampler_sex_classification, sampler_UKB_scan_age, sampler_HCP_fluid_intelligence

import torch

##### METRICS DATALOADER ########

def loader_metrics(data_path,
                    sampler,
                    config,
                    split_cv=-1):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################


    if sampler and config['data']['dataset']=='dHCP' and config['data']['low_train']:
        print('Loading partial train set: {}%'.format(config['data']['low_train']))
        train_id = 'train_{}'.format(config['data']['low_train'])
    else:
        train_id = 'train'

    train_dataset = dataset_cortical_surfaces(config=config,
                                                data_path=data_path,
                                                split=train_id,
                                                split_cv=split_cv)

    #####################################
    ###############  dHCP  ##############
    #####################################
    if sampler and config['data']['dataset']=='dHCP' :
        
        if config['data']['task']=='birth_age' :
            print('Sampler: dHCP preterm for birth_age')

            if config['data']['low_train']:
                train_id = 'train_{}'.format(config['data']['low_train'])
            else:
                train_id = 'train'

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],train_id)).labels.to_numpy()
            sampler = sampler_preterm_birth_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
        elif config['data']['task']=='scan_age':
            print('Sampler: dHCP preterm for scan_age')

            if config['data']['low_train']:
                train_id = 'train_{}'.format(config['data']['low_train'])
            else:
                train_id = 'train'

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],train_id)).labels.to_numpy()
            sampler = sampler_preterm_scan_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
            
        elif config['data']['task']=='sex':
            print('Sampler: dHCP sex classification')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_sex_classification(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)
                                            
    #####################################
    ###############  UKB   ##############
    #####################################
    elif sampler and config['data']['dataset']=='UKB':
    
        if config['data']['task']=='sex' or config['data']['task']=='sex_msmall':
            print('Sampler: UKB sex classification')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_sex_classification(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)

        elif config['data']['task']=='scan_age':
            print('Sampler: UKB scan age regression')

            train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                        config['data']['dataset'],
                                                                                        config['data']['task'],
                                                                                        config['data']['hemi'],'train')).labels.to_numpy()
            sampler = sampler_UKB_scan_age(train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32)

    #####################################
    ###############  HCP   ##############
    #####################################
    elif sampler and config['data']['dataset']=='HCP':
        
        if config['training']['use_cross_validation']: 
        
            if config['data']['task']=='iq':

                print('Sampler: HCP fluid intelligence regression')

                train_labels = pd.read_csv('{}/labels/{}/{}/cv_5/{}/{}/train{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['modality'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],
                                                                                            split_cv)).labels.to_numpy()
                sampler = new_sampler_HCP_fluid_intelligence(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
            if config['data']['task']=='sex':

                print('Sampler: HCP fluid intelligence regression')

                train_labels = pd.read_csv('{}/labels/{}/{}/cv_5/{}/{}/train{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['modality'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],
                                                                                            split_cv)).labels.to_numpy()
                sampler = sampler_sex_classification(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
            
            
        # non cross validation
        else:
            if config['data']['task']=='sex':

                print('Sampler: HCP sex classification')

                train_labels = pd.read_csv('{}/labels/{}/cortical_metrics/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                                            config['data']['dataset'],
                                                                                            config['data']['task'],
                                                                                            config['data']['hemi'],'train')).labels.to_numpy()
                sampler = sampler_sex_classification(train_labels)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = config['training']['bs'],
                                                        sampler=sampler,
                                                        num_workers=32,)
    else:
        print('not using sampler...')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = config['training']['bs'],
                                                    shuffle=(not config['RECONSTRUCTION']),
                                                    num_workers=32)

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    if split_cv == -1:

        val_dataset = dataset_cortical_surfaces(data_path=data_path,
                                                    config=config,
                                                    split='val',
                                                    split_cv=split_cv)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['training']['bs_val'],
                                                shuffle=False,
                                                num_workers=32)
                

    ###############################################################
    #####################    TESTING DATA     #####################
    ###############################################################
            

    test_dataset = dataset_cortical_surfaces(data_path=data_path,
                                            config=config,
                                            split='test',
                                            split_cv=split_cv)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False, 
                                            num_workers=32)
    
    train_dataset.logging()
    
    if split_cv ==-1:

        print('')
        print('#'*30)
        print('############ Data ############')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Validation data: {}'.format(len(val_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, val_loader, test_loader
    
    else:

        print('')
        print('#'*30)
        print('########### Data ###########')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, test_loader

def loader_tfmri(data_path,
                config,
                split_cv=-1):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_cortical_surfaces_tfmri(config=config,
                                                    data_path=data_path,
                                                    split='train',
                                                    split_cv=split_cv)


    #####################################
    ###############  HCP   ##############
    #####################################
    if config['data']['dataset']=='HCP':
        
        print('not using sampler...')
        print('shuffling == {}'.format(not config['RECONSTRUCTION']))


        assert config['training']['bs']%config['fMRI']['nbr_clip_sampled_fmri']==0
        batch_size_training = config['training']['bs']//config['fMRI']['nbr_clip_sampled_fmri']
        print('batch size training: {}'.format(batch_size_training))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size_training,
                                                    shuffle=(not config['RECONSTRUCTION']),
                                                    num_workers=18,
                                                    pin_memory=True)
    

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    if split_cv == -1:

        val_dataset = dataset_cortical_surfaces_tfmri(data_path=data_path,
                                                    config=config,
                                                    split='val',
                                                    split_cv=split_cv)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['training']['bs_val'],
                                                shuffle=False,
                                                num_workers=32)
                
    print('')
    print('#'*30)
    print('############ Data ############')
    print('#'*30)
    print('')

    print('')
    print('Training data: {}'.format(len(train_dataset)))
    print('Validation data: {}'.format(len(val_dataset)))
        
    return train_loader, val_loader

def loader_rfmri(data_path,
                config,
                split_cv=-1):

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_cortical_surfaces_rfmri(config=config,
                                                    data_path=data_path,
                                                    split='train',
                                                    split_cv=split_cv)


    #####################################
    ###############  HCP   ##############
    #####################################
    if config['data']['dataset']=='HCP':
        
        print('not using sampler...')
        print('shuffling == {}'.format(not config['RECONSTRUCTION']))


        assert config['training']['bs']%config['fMRI']['nbr_clip_sampled_fmri']==0
        batch_size_training = config['training']['bs']//config['fMRI']['nbr_clip_sampled_fmri']
        print('batch size training: {}'.format(batch_size_training))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size_training,
                                                    shuffle=(not config['RECONSTRUCTION']),
                                                    num_workers=0,
                                                    pin_memory=True)
    

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    if split_cv == -1:

        val_dataset = dataset_cortical_surfaces_rfmri(data_path=data_path,
                                                    config=config,
                                                    split='val',
                                                    split_cv=split_cv)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config['training']['bs_val'],
                                                shuffle=False,
                                                num_workers=32)
                
    print('')
    print('#'*30)
    print('############ Data ############')
    print('#'*30)
    print('')

    print('')
    print('Training data: {}'.format(len(train_dataset)))
    print('Validation data: {}'.format(len(val_dataset)))
        
    return train_loader, val_loader

    ###############################################################
    #####################    TESTING DATA     #####################
    ###############################################################
            

    test_dataset = dataset_cortical_surfaces_fmri(data_path=data_path,
                                            config=config,
                                            split='test',
                                            split_cv=split_cv)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config['training']['bs_val'],
                                            shuffle=False, 
                                            num_workers=32)
    
    train_dataset.logging()
    
    if split_cv ==-1:

        print('')
        print('#'*30)
        print('############ Data ############')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Validation data: {}'.format(len(val_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, val_loader, test_loader
    
    else:

        print('')
        print('#'*30)
        print('########### Data ###########')
        print('#'*30)
        print('')

        print('')
        print('Training data: {}'.format(len(train_dataset)))
        print('Testing data: {}'.format(len(test_dataset)))

        return train_loader, test_loader
