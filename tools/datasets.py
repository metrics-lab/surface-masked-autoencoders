from logging import raiseExceptions
import os
import sys 
sys.path.append('./')
sys.path.append('../')

import numpy as np
import nibabel as nb
import pandas as pd
import random 

import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R

from surfaces.metric_resample import *
from surfaces.metric_resample_labels import *
from skimage.exposure import match_histograms


from einops import rearrange, reduce

import warnings
import torchaudio
from samplers import balance_dataset

    
class dataset_cortical_surfaces(Dataset):
    def __init__(self, 
                data_path,
                config,
                split,
                split_cv=-1,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        task = config['data']['task']
        sampling = config['mesh_resolution']['sampling']
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']
        
        self.filedir =  data_path
        self.split = split
        self.configuration = config['data']['configuration']
        self.dataset = config['data']['dataset']
        self.path_to_workdir = config['data']['path_to_workdir']
        self.hemi = config['data']['hemi']
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.normalise = config['data']['normalise']
        self.use_confounds = config['training']['use_confounds']
        self.use_cross_val = config['training']['use_cross_validation']
        self.clipping = config['data']['clipping']
        self.modality = config['data']['modality']
        self.path_to_template = config['data']['path_to_template']
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['ico_{}_grid'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['ico_{}_grid'.format(sub_ico)]['num_patches']
        self.data_harmonisation= config['data']['data_harmonisation']

        if self.data_harmonisation:
            self.mean_channel_1_ukb = np.load('/home/sd20/workspace/sMAE/labels/UKB/cortical_metrics/scan_age/half/mean_0.npy')
            self.mean_channel_2_ukb = np.load('/home/sd20/workspace/sMAE/labels/UKB/cortical_metrics/scan_age/half/mean_1.npy')
            self.mean_channel_3_ukb = np.load('/home/sd20/workspace/sMAE/labels/UKB/cortical_metrics/scan_age/half/mean_2.npy')
            self.mean_channel_4_ukb = np.load('/home/sd20/workspace/sMAE/labels/UKB/cortical_metrics/scan_age/half/mean_3.npy')
        
        if config['MODEL'] == 'sit' or config['MODEL']=='ms-sit':
            self.patching=True
            self.channels = config['transformer']['channels']
            self.num_channels = len(self.channels)
                
        elif config['MODEL']== 'spherical-unet':
            self.patching=False 
            self.channels = config['spherical-unet']['channels']
            self.num_channels = len(self.channels)

        else:
            raiseExceptions('model not implemented yet')
        
        ################################################
        ##############       LABELS       ##############
        ################################################

        if self.use_cross_val and split_cv>=0:
            self.data_info = pd.read_csv('{}/labels/{}/{}/cv_5/{}/{}/{}{}.csv'.format(config['data']['path_to_workdir'],
                                                                                    config['data']['dataset'],
                                                                                    config['data']['modality'],
                                                                                    config['data']['task'],
                                                                                    config['data']['hemi'],
                                                                                    split,
                                                                                    split_cv))
        else:                                                                     
            self.data_info = pd.read_csv('{}/labels/{}/{}/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                            self.dataset,self.modality,task,
                                                                             self.hemi,
                                                                             split))
        self.filenames = self.data_info['ids']
        self.labels = self.data_info['labels']
        if self.use_confounds: 
            self.confounds = self.data_info['confounds']
        else:
            self.confounds= False
        
        ###################################################
        ##############       NORMALISE       ##############
        ###################################################

        if self.normalise=='group-standardise' and self.use_cross_val:
        
            self.means = np.load('{}/labels/{}/cortical_metrics/cv_5/{}/{}/{}/means.npy'.format(config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
            self.stds = np.load('{}/labels/{}/cortical_metrics/cv_5/{}/{}/{}/stds.npy'.format(self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
                                                                            
        elif self.normalise=='group-standardise' and not self.use_cross_val:
        
            self.means = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/means.npy'.format(config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
            self.stds = np.load('{}/labels/{}/cortical_metrics/{}/{}/{}/stds.npy'.format(config['data']['path_to_workdir'],
                                                                                self.dataset,task,
                                                                                self.hemi,
                                                                                self.configuration))
        

        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################
        
        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],sampling,ico,sub_ico))

        #config, augmentation
        if self.augmentation:
            self.rotation = config['augmentation']['prob_rotation']
            self.max_degree_rot = config['augmentation']['max_abs_deg_rotation']
            self.shuffle = config['augmentation']['prob_shuffle']
            self.warp = config['augmentation']['prob_warping']
            self.coord_ico6 = np.array(nb.load('{}/coordinates_ico_6_L.shape.gii'.format(self.path_to_template)).agg_data()).T        

        if config['mesh_resolution']['reorder']:
            #reorder patches 
            new_order_indices = np.load('{}/patch_extraction/order_patches/order_ico{}.npy'.format(config['data']['path_to_workdir'],sub_ico))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.triangle_indices.columns))}
            self.triangle_indices = self.triangle_indices[list([str(i) for i in new_order_indices])]
            self.triangle_indices = self.triangle_indices.rename(columns=d)
        
        self.masking = config['data']['masking']    
        if self.masking and self.dataset == 'dHCP' and self.configuration == 'template' : # for dHCP
            if split == 'train':
                print('Masking the cut: dHCP mask')
            self.mask = np.array(nb.load('{}/week-40_hemi-left_space-dhcpSym_dens-40k_desc-medialwallsymm_mask.shape.gii'.format(self.path_to_template)).agg_data())
        elif self.masking and self.dataset == 'dHCP' and self.configuration == 'native':
            if split == 'train':
                print('Masking the cut: dHCP - using native subject masks')
        elif self.masking and self.dataset == 'UKB': # for UKB
            if split == 'train':
                print('Masking the cut: UKB mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.ico6_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        elif self.masking and self.dataset == 'HCP': # for HCP
            if split == 'train':
                print('Masking the cut: HCP mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        else:
            if split == 'train':
                print('Masking the cut: NO')
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        ############
        # 1. load input metric
        # 2. select input channels
        # 3. mask input data (if masking)
        # 4. clip input data (if clipping)
        # 5. remask input data (if masking)
        # 6. normalise data (if normalise)
        # 7. apply augmentation
        # 8. get sequence of patches
        ############

        ### label
        label = self.labels.iloc[idx]
        if self.use_confounds:
            confound = self.confounds.iloc[idx]
            label = np.array([label, confound])

        ### hemisphere
        if self.hemi == 'half':

            data = self.get_half_hemi(idx)

            if (self.augmentation and self.split =='train'):
                # do augmentation or not do augmentation
                if np.random.rand() > (1-self.augmentation):
                    # chose which augmentation technique to use
                    p = np.random.rand()
                    if p < self.rotation:
                        #apply rotation
                        data = self.apply_rotation(data)

                    elif self.rotation <= p < self.rotation+self.warp:
                        #apply warp
                        data = self.apply_non_linear_warp(data)

                    elif self.rotation+self.war <= p < self.rotation+self.warp+self.shuffle:
                        #apply shuffle
                        data = self.apply_shuffle(data)
                        
            if self.patching:

                sequence = self.get_sequence(data)

                if self.augmentation and self.shuffle and self.split == 'train':
                    bool_select_patch =np.random.rand(sequence.shape[1])<self.prob_shuffle
                    copy_data = sequence[:,bool_select_patch,:]
                    ind = np.where(bool_select_patch)[0]
                    np.random.shuffle(ind)  
                    sequence[:,ind,:]=copy_data
                return (torch.from_numpy(sequence).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

            else:
                return (torch.from_numpy(data).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

        return (torch.from_numpy(sequence).float(),torch.from_numpy(np.asarray(label,dtype=float)).float())

    
    def get_half_hemi(self,idx):

        #### 1. masking
        #### 2. normalising - only vertices that are not masked

        path = os.path.join(self.filedir,self.filenames.iloc[idx])
        data =  np.array(nb.load(path).agg_data())

        if self.data_harmonisation:
            data[0] = match_histograms(data[0], self.mean_channel_1_ukb, channel_axis=None)
            data[1] = match_histograms(data[1], self.mean_channel_2_ukb, channel_axis=None)
            data[2] = match_histograms(data[2], self.mean_channel_3_ukb, channel_axis=None)
            data[3] = match_histograms(data[3], self.mean_channel_4_ukb, channel_axis=None)
            data = np.array(data)


        if len(data.shape)==1:
            data = np.expand_dims(data,0)
        data = data[self.channels,:]

        # load individual mask if native
        if self.masking and self.dataset=='dHCP' and self.configuration=='native':
            self.mask = np.array(nb.load(os.path.join(self.filedir, 'native_masks','{}.medialwall_mask.shape.gii'.format(self.filenames.iloc[idx].split('.')[0]))).agg_data())

        if self.masking:
            data = np.multiply(data,self.mask)
            if self.clipping:
                data = self.clipping_(data)
            data = np.multiply(data,self.mask) ## need a second masking to remove the artefacts after clipping
        else:
            if self.clipping:
                data = self.clipping_(data)

        data = self.normalise_(data)

        return data

    def clipping_(self,data):

        if self.dataset == 'dHCP':
            if self.data_harmonisation:
                lower_bounds = np.array([0.0,-0.41, -0.40, -16.50])
                upper_bounds = np.array([2.41, 0.45,5.33, 15.4])
                for i,channel in enumerate(self.channels):
                    data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])
            else:
                lower_bounds = np.array([0.0,-0.5, -0.05, -10.0])
                upper_bounds = np.array([2.2, 0.6,2.6, 10.0 ])
                for i,channel in enumerate(self.channels):
                    data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])
        elif self.dataset == 'UKB':
            lower_bounds = np.array([0.0,-0.41, -0.40, -16.50])
            upper_bounds = np.array([2.41, 0.45,5.33, 15.4])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        elif self.dataset == 'HCP' and self.modality == 'memory_task':
            lower_bounds = np.array([-10.86,-9.46,-6.39])
            upper_bounds = np.array([12.48,10.82,6.68])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        elif self.dataset == 'HCP' and self.modality == 'cortical_metrics':
            lower_bounds = np.array([-0.01,-0.41,-0.23,-1.65])
            upper_bounds = np.array([2.44,0.46,5.09,1.52])
            for i,channel in enumerate(self.channels):
                data[i,:] = np.clip(data[i,:], lower_bounds[channel], upper_bounds[channel])

        return data

    def normalise_(self,data):

        if self.masking:
            non_masked_vertices = self.mask>0
            if self.normalise=='group-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - self.means[:,self.channels,:].reshape(self.num_channels,1))/self.stds[:,self.channels,:].reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(self.num_channels,1))/data[:,non_masked_vertices].std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='sub-normalise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))/(data[:,non_masked_vertices].max(axis=1).reshape(self.num_channels,1)- data[:,non_masked_vertices].min(axis=1).reshape(self.num_channels,1))
        
        else:
            if self.normalise=='group-standardise':
                data= (data- self.means.reshape(self.num_channels,1))/self.stds.reshape(self.num_channels,1)
            elif self.normalise=='sub-standardise':
                data = (data - data.mean(axis=1).reshape(self.num_channels,1))/data.std(axis=1).reshape(self.num_channels,1)
            elif self.normalise=='normalise':
                data = (data- data.min(axis=1).reshape(self.num_channels,1))/(data.max(axis=1).reshape(self.num_channels,1)- data.min(axis=1).reshape(self.num_channels,1))
        return data
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.num_channels, self.nbr_patches, self.nbr_vertices))
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence

    def apply_rotation(self,data):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
        axis = random.choice(['x','y','z'])

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, 'cpu')

        return rotated_moving_img.numpy().T

    def apply_non_linear_warp(self,data):

        # chose one warps at random between 0 - 99

        id = np.random.randint(0,100)
        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico, id)).agg_data()
        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, 'cpu')
        return warped_moving_img.numpy().T
    
    ############ LOGGING ############

    def logging(self):
        
        if self.split == 'train':
                print('Using {} channels'.format(self.channels))
        
        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     - rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     - rotations: no')
            if self.shuffle:
                print('     - shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     - shuffling: no')
            if self.warp:
                print('     - non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     - non-linear warping: no')
        else:
            print('Augmentation: NO')

class dataset_cortical_surfaces_tfmri(Dataset):
    def __init__(self, 
                data_path,
                config,
                split,
                split_cv=-1,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        self.task = config['data']['task']
        sampling = config['mesh_resolution']['sampling']
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']
        self.balance = config['data']['balance']
        
        self.filedir =  data_path
        self.split = split
        self.configuration = config['data']['configuration']
        self.dataset = config['data']['dataset']
        self.path_to_workdir = config['data']['path_to_workdir']
        self.hemi = config['data']['hemi']
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.normalise = config['data']['normalise']
        self.use_confounds = config['training']['use_confounds']
        self.use_cross_val = config['training']['use_cross_validation']
        self.modality = config['data']['modality']
        self.path_to_template = config['data']['path_to_template']
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['ico_{}_grid'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['ico_{}_grid'.format(sub_ico)]['num_patches']
        self.masking = config['data']['masking']
        self.demean = config['data']['demean']

        ### fmri
        self.nbr_frames = config['fMRI']['nbr_frames']
        self.temporal_window = config['fMRI']['window']
        self.sampling_type = config['fMRI']['sampling_type']
        self.temporal_rep = config['fMRI']['temporal_rep']
        self.temporal_lag = config['fMRI']['temporal_lag']
        self.nbr_clip_to_sample = config['fMRI']['nbr_clip_sampled_fmri'] if split == 'train' else 1
        self.nbr_frames_to_extract = config['fMRI']['nbr_frames']

        assert config['training']['bs']%self.nbr_clip_to_sample==0
        
        if config['MODEL'] == 'sit':
            self.patching=True
            self.channels = [0] # by default
            self.num_channels = len(self.channels)

        else:
            raiseExceptions('model not implemented yet')

        ################################################
        ##############       LABELS       ##############
        ################################################

        self.split_info = pd.read_csv('{}/labels/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                              self.dataset,self.modality,
                                                                                split))
                
        self.subject_ids = self.split_info['ids']
        self.subject_movies = self.split_info['movie']
        self.subject_sessions = self.split_info['session']
        self.subject_hemi = self.split_info['hemi']

        #print(self.subject_ids , self.subject_movies , self.subject_sessions , self.subject_hemi )

        if self.use_confounds: 
            self.confounds = self.split_info['confounds']
        else:
            self.confounds= False


        if self.task!='None':
            self.task_info = pd.read_csv('{}/labels/{}/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                            self.dataset,self.modality,self.task,
                                                                             self.task))
            #print(self.task_info)
            
            
        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################
        
        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],sampling,ico,sub_ico))    
        print('ico res: {}'.format(sub_ico))

        if config['mesh_resolution']['reorder']:
            #reorder patches 
            new_order_indices = np.load('{}/patch_extraction/order_patches/order_ico{}.npy'.format(config['data']['path_to_workdir'],sub_ico))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.triangle_indices.columns))}
            self.triangle_indices = self.triangle_indices[list([str(i) for i in new_order_indices])]
            self.triangle_indices = self.triangle_indices.rename(columns=d)
        
        if self.masking and self.dataset == 'HCP': # for HCP
            if split == 'train':
                print('Masking the cut: HCP mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        
            
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self,idx):
        ############
        # 1. load input bold timeseries
        # 2. select input channels
        # 3. mask input data (if masking)
        # 4. clip input data (if clipping)
        # 5. remask input data (if masking)
        # 6. normalise data (if normalise)
        # 7. apply augmentation
        # 8. get sequence of patches
        ############

        ### label
        #label = self.labels.iloc[idx]
        #if self.use_confounds:
        #    confound = self.confounds.iloc[idx]
        #    label = np.array([label, confound])

        ### hemisphere
        if self.hemi == 'half':

            if self.task!='None':

                #get information for subject idx
                subject_info = self.split_info.loc[idx]
                #select only the subset from the task info from the corresponding movie
                task_info = self.task_info[self.task_info['movie']==subject_info.movie]
                #print(task_info)
                if self.task!='None' and self.balance:
                    task_info = balance_dataset(task_info)

                #sample one of the frame at random
                row = task_info.sample(n=1)
                #get only the selected movie
                frame = row['frame'].values[0]
                label = row['labels'].values
                nbr_frames = row['nbr_frames']
                
                frames_to_extract = self.extract_clip(nbr_frames,t_0=frame)

                input_data = []
                for f in frames_to_extract:
                    input_data.append(self.get_half_hemi(idx,f))

                input_data = np.stack(input_data,axis=0)

                input_data = self.normalise_(input_data)
                            
                if self.patching:

                    sequence = self.get_sequence(data)
                    if self.nbr_frames_to_extract > 1:
                        sequence = self.temporal_mixing(sequence)
                                            
                return (torch.from_numpy(sequence).float(),
                        torch.from_numpy(np.asarray(label,dtype=float)).float())

            else: 

                #get information for subject idx
                subject_info = self.split_info.loc[idx]
                #select only the subset from the task info from the corresponding movie

                nbr_frames = subject_info['nbr_frames']

                #sample one of the frame at random
                frame = random.randint(0,nbr_frames-1)

                frames_to_extract = self.extract_clip(nbr_frames,t_0=frame)

                input_data = []
                for f in frames_to_extract:
                    input_data.append(self.get_half_hemi(idx,f.cpu().numpy()))

                input_data = np.stack(input_data,axis=0)

                input_data = self.normalise_(input_data)

                if self.patching:

                    sequence = self.get_sequence(input_data)
                    if self.nbr_frames_to_extract > 1:
                        sequence = self.temporal_mixing(sequence)
                    
            return torch.from_numpy(sequence).float()

                    
    
    def get_half_hemi(self,idx, frame):

        #### 1. masking
        #### 2. normalising - only vertices that are not masked

        path = os.path.join(self.filedir,
                            str(self.subject_ids.iloc[idx]),
                            'tfMRI_MOVIE{}_7T_{}'.format(self.subject_movies[idx],self.subject_sessions[idx]),
                            'frames_ico6_{}'.format(self.subject_hemi[idx]),
                            'frame_{}.{}.shape.gii'.format(str(frame).zfill(3),self.subject_hemi[idx]))
        data =  np.array(nb.load(path).agg_data())
        
        if len(data.shape)==1:
            data = np.expand_dims(data,0)

        return data.squeeze()

    def demean_(self,data):
        
        if self.demean:
            data = (data - data.mean(axis=0).reshape(1,-1))
            
        return data
    
    def normalise_(self,data):

        if self.masking:
            non_masked_vertices = self.mask>0
            if self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(data.shape[0],1))/data[:,non_masked_vertices].std(axis=1).reshape(data.shape[0],1)
        
        return data

    def extract_clip(self,nbr_frames,t_0=None):

        if t_0==None:
            t_0 = torch.randint(low = 0, high= self.video_length-self.temporal_window,size=(1,)).item()
        #print('******* sampling type - {} *******:'.format(self.sampling_type))

        t_0 = min(t_0, t_0+self.temporal_lag)

        if self.sampling_type == 'uniform':
            #from T0 to T0+T uniformly
            #print(t_0 , min(t_0+self.temporal_window,fmri_vid.shape[0])-1)
            frames_to_extract = torch.round(torch.linspace(t_0,min(t_0+self.temporal_window,nbr_frames)-1,self.nbr_frames),).int()

        elif self.sampling_type == 'chunk':
            #sampling a chunk of frames from T0+T0'
            t_0_prime = torch.randint(low = t_0, high=t_0 + self.temporal_window - self.nbr_frames-1,size=(1,)).item()
            frames_to_extract = [t_0_prime+i for i in range(self.nbr_frames)]

        elif self.sampling_type == 'random':
            #sampling random frames starting from T0
            frames_to_extract = torch.cat((torch.tensor([t_0]),torch.randperm(self.temporal_window-1)[:self.nbr_frames-1].sort()[0]+min(t_0, nbr_frames-self.temporal_window)+1))
        else:
            raise('Not implemented yet')
        
        return frames_to_extract
    
    def temporal_mixing(self,sequence):

        #print('******* temporal mixing - {} *******:'.format(self.temporal_rep))

        if self.temporal_rep == 'concat':

            #print('sequence: {}'.format(sequence.shape))
            #### Not sure which option to chose, look at smae_video_dev.ipynb
            #sequence = rearrange(sequence, 't n v -> 1 (n t) v') #concat patches
            sequence = rearrange(sequence, 't n v -> 1 (t n) v') #concat frames
            #print('sequence: {}'.format(sequence.shape))
        
        elif self.temporal_rep == 'avg':

            #print('sequence: {}'.format(sequence.shape))
            sequence = reduce(sequence, 't n v -> 1 n v', 'mean')
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'mix':

            #print('sequence: {}'.format(sequence.shape))
            mask = np.eye(self.nbr_frames,dtype=bool)[:, np.random.choice(self.nbr_frames, self.nbr_patches)]
            sequence = sequence[mask][np.newaxis,:,:]
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'tubelet':

            print('sequence: {}'.format(sequence.shape))

        else:
            raise('Not implemented yet')
        
        return sequence
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.nbr_frames, self.nbr_patches, self.nbr_vertices))
        #print(sequence.shape)
        #print(data.shape)
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence

    def apply_rotation(self,data):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
        axis = random.choice(['x','y','z'])

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, 'cpu')

        return rotated_moving_img.numpy().T

    def apply_non_linear_warp(self,data):

        # chose one warps at random between 0 - 99

        id = np.random.randint(0,100)
        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico, id)).agg_data()
        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, 'cpu')
        return warped_moving_img.numpy().T
    
    ############ LOGGING ############

    def logging(self):
        
        if self.split == 'train':
                print('Using {} channels'.format(self.channels))
        
        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     - rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     - rotations: no')
            if self.shuffle:
                print('     - shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     - shuffling: no')
            if self.warp:
                print('     - non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     - non-linear warping: no')
        else:
            print('Augmentation: NO')

class dataset_cortical_surfaces_rfmri(Dataset):
    def __init__(self, 
                data_path,
                config,
                split,
                split_cv=-1,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        self.task = config['data']['task']
        sampling = config['mesh_resolution']['sampling']
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']
        self.balance = config['data']['balance']
        
        self.filedir =  data_path
        self.split = split
        self.configuration = config['data']['configuration']
        self.dataset = config['data']['dataset']
        self.path_to_workdir = config['data']['path_to_workdir']
        self.hemi = config['data']['hemi']
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.normalise = config['data']['normalise']
        self.use_confounds = config['training']['use_confounds']
        self.use_cross_val = config['training']['use_cross_validation']
        self.modality = config['data']['modality']
        self.path_to_template = config['data']['path_to_template']
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['ico_{}_grid'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['ico_{}_grid'.format(sub_ico)]['num_patches']
        self.masking = config['data']['masking']
        self.demean = config['data']['demean']
        self.subset = config['data']['subset']
        self.nbr_frames_to_extract = config['fMRI']['nbr_frames']

        ### fmri
        self.nbr_frames = config['fMRI']['nbr_frames']
        self.temporal_window = config['fMRI']['window']
        self.sampling_type = config['fMRI']['sampling_type']
        self.temporal_rep = config['fMRI']['temporal_rep']
        self.temporal_lag = config['fMRI']['temporal_lag']
        self.nbr_clip_to_sample = config['fMRI']['nbr_clip_sampled_fmri'] if split == 'train' else 1

        assert config['training']['bs']%self.nbr_clip_to_sample==0
        
        if config['MODEL'] == 'sit':
            self.patching=True
            self.channels = [0] # by default
            self.num_channels = len(self.channels)

        else:
            raiseExceptions('model not implemented yet')

        ################################################
        ##############       LABELS       ##############
        ################################################

        if self.subset:
            self.split_info = pd.read_csv('{}/labels/{}/{}/subset/{}.csv'.format(config['data']['path_to_workdir'],
                                                                              self.dataset,self.modality,
                                                                                split))
        else:
            self.split_info = pd.read_csv('{}/labels/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                              self.dataset,self.modality,
                                                                                split))
                        
        self.subject_ids = self.split_info['ids']
        self.subject_rest = self.split_info['rest']
        self.subject_sessions = self.split_info['session']
        self.subject_hemi = self.split_info['hemi']

        if self.use_confounds: 
            self.confounds = self.split_info['confounds']
        else:
            self.confounds= False


        if self.task!='None':
            self.task_info = pd.read_csv('{}/labels/{}/{}/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                            self.dataset,self.modality,self.task,
                                                                             self.task))            
            
        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################
        
        self.triangle_indices = pd.read_csv('{}/patch_extraction/{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],sampling,ico,sub_ico))    
        print('ico res: {}'.format(sub_ico))

        if config['mesh_resolution']['reorder']:
            #reorder patches 
            new_order_indices = np.load('{}/patch_extraction/order_patches/order_ico{}.npy'.format(config['data']['path_to_workdir'],sub_ico))
            d = {str(new_order_indices[i]):str(i) for i in range(len(self.triangle_indices.columns))}
            self.triangle_indices = self.triangle_indices[list([str(i) for i in new_order_indices])]
            self.triangle_indices = self.triangle_indices.rename(columns=d)
        
        if self.masking and self.dataset == 'HCP': # for HCP
            if split == 'train':
                print('Masking the cut: HCP mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())
        
            
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self,idx):
        ############
        # 1. load input bold timeseries
        # 2. select input channels
        # 3. mask input data (if masking)
        # 4. clip input data (if clipping)
        # 5. remask input data (if masking)
        # 6. normalise data (if normalise)
        # 7. apply augmentation
        # 8. get sequence of patches
        ############

        ### hemisphere
        if self.hemi == 'half':

            #get information for subject idx
            subject_info = self.split_info.loc[idx]
            #select only the subset from the task info from the corresponding movie

            nbr_frames = subject_info['nbr_frames']

            #sample one of the frame at random
            frame = random.randint(0,nbr_frames-1)

            frames_to_extract = self.extract_clip(nbr_frames,t_0=frame)

            input_data = []
            for f in frames_to_extract:
                input_data.append(self.get_half_hemi(idx,f.cpu().numpy()))

            input_data = np.stack(input_data,axis=0)

            input_data = self.normalise_(input_data)
                                
            if self.patching:

                sequence = self.get_sequence(input_data)
                if self.nbr_frames_to_extract > 1:
                    sequence = self.temporal_mixing(sequence)
                    
            return torch.from_numpy(sequence).float()


    def get_half_hemi(self,idx,frame):

        path = os.path.join(self.filedir,
                            str(self.subject_ids.iloc[idx]),
                            'rfMRI_REST{}_7T_{}'.format(self.subject_rest[idx],self.subject_sessions[idx]),
                            'frames_ico6_{}'.format(self.subject_hemi[idx]),
                            'frame_{}.{}.shape.gii'.format(str(frame).zfill(3),self.subject_hemi[idx]))
        #print(path)
        data =  np.array(nb.load(path).agg_data())
        
        if len(data.shape)==1:
            data = np.expand_dims(data,0)

        return data.squeeze()

    def demean_(self,data):
        
        if self.demean:
            data = (data - data.mean(axis=0).reshape(1,-1))
            
        return data
    
    def normalise_(self,data):

        if self.masking:

            #print('Normalise: {}'.format(self.normalise))
            
            non_masked_vertices = self.mask>0
            #print(data.shape)
            #print(data[:,non_masked_vertices].mean(axis=1).shape)
            #print(data[:,non_masked_vertices].std(axis=1).shape)
            if self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(data.shape[0],1))/data[:,non_masked_vertices].std(axis=1).reshape(data.shape[0],1)
            elif self.normalise=='sub-normalise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].min(axis=1).reshape(data.shape[0],1))/(data[:,non_masked_vertices].max(axis=1).reshape(data.shape[0],1)- data[:,non_masked_vertices].min(axis=1).reshape(data.shape[0],1))
        
        
        return data
    
    def extract_clip(self,nbr_fames,t_0=None):

        if t_0==None:
            t_0 = torch.randint(low = 0, high= self.video_length-self.temporal_window,size=(1,)).item()

        t_0 = min(t_0, t_0+self.temporal_lag) ### why ??

        if self.sampling_type == 'uniform':
            #from T0 to T0+T uniformly
            print(t_0 , min(t_0+self.temporal_window,nbr_fames)-1)
            frames_to_extract = torch.round(torch.linspace(t_0,min(t_0+self.temporal_window,nbr_fames)-1,self.nbr_frames_to_extract),).int()
            print(frames_to_extract)

        elif self.sampling_type == 'chunk':
            #sampling a chunk of frames from T0+T0'
            t_0_prime = torch.randint(low = t_0, high=t_0 + self.temporal_window - self.nbr_frames_to_extract-1,size=(1,)).item()
            frames_to_extract = [t_0_prime+i for i in range(self.nbr_frames_to_extract)]

        elif self.sampling_type == 'random':
            #sampling random frames starting from T0
            frames_to_extract = torch.cat((torch.tensor([t_0]),torch.randperm(self.temporal_window-1)[:self.nbr_frames_to_extract-1].sort()[0]+min(t_0, nbr_fames-self.temporal_window)+1))
        else:
            raise('Not implemented yet')
        
        return frames_to_extract

    
    def temporal_mixing(self,sequence):

        #print('******* temporal mixing - {} *******:'.format(self.temporal_rep))

        if self.temporal_rep == 'concat':

            #print('sequence: {}'.format(sequence.shape))
            #### Not sure which option to chose, look at smae_video_dev.ipynb
            #sequence = rearrange(sequence, 't n v -> 1 (n t) v') #concat patches
            sequence = rearrange(sequence, 't n v -> 1 (t n) v') #concat frames
            #print('sequence: {}'.format(sequence.shape))
        
        elif self.temporal_rep == 'avg':

            #print('sequence: {}'.format(sequence.shape))
            sequence = reduce(sequence, 't n v -> 1 n v', 'mean')
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'mix':

            #print('sequence: {}'.format(sequence.shape))
            mask = np.eye(self.nbr_frames,dtype=bool)[:, np.random.choice(self.nbr_frames, self.nbr_patches)]
            sequence = sequence[mask][np.newaxis,:,:]
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'tubelet':

            print('sequence: {}'.format(sequence.shape))

        else:
            raise('Not implemented yet')
        
        return sequence
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.nbr_frames, self.nbr_patches, self.nbr_vertices))
        #import pdb;pdb.set_trace()
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence

    def apply_rotation(self,data):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
        axis = random.choice(['x','y','z'])

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, 'cpu')

        return rotated_moving_img.numpy().T

    def apply_non_linear_warp(self,data):

        # chose one warps at random between 0 - 99

        id = np.random.randint(0,100)
        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico, id)).agg_data()

        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, 'cpu')
        
        return warped_moving_img.numpy().T
    
    ############ LOGGING ############

    def logging(self):
        
        if self.split == 'train':
                print('Using {} channels'.format(self.channels))
        
        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     - rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     - rotations: no')
            if self.shuffle:
                print('     - shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     - shuffling: no')
            if self.warp:
                print('     - non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     - non-linear warping: no')
        else:
            print('Augmentation: NO')

