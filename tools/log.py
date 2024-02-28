import os
import numpy as np
import torch
from scipy.stats import pearsonr
from tools.utils import plot_regression_results_UKB, plot_regression_results_HCP
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, r2_score
from tools.utils import save_reconstruction_mae,save_reconstruction_mae_fmri
import yaml

def tensorboard_log_train(config, writer, scheduler, optimizer, loss_item, iter_count):
    
    writer.add_scalar('loss/train_it', loss_item, iter_count)

    if config['optimisation']['use_scheduler']:
        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            if config['optimisation']['warmup']:
                if (iter_count)<config['optimisation']['nbr_step_warmup']:
                    writer.add_scalar('LR',scheduler.get_lr()[0], iter_count)
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
            else:
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
        else:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    else:
        if config['optimisation']['warmup']:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'],iter_count )
        else:
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    
    return scheduler, writer

def tensorboard_log_pretrain_trainset(config, writer, scheduler, optimizer, loss_item, iter_count):
    
    writer.add_scalar('loss/train_it', loss_item,  iter_count)

    if config['optimisation']['use_scheduler']:
        scheduler.step()
        writer.add_scalar('LR',optimizer.param_groups[0]['lr'],  iter_count)
    else:
        if config['optimisation']['warmup']:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
        else:
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'],  iter_count)
            
    return scheduler, writer

def tensorboard_log_pretrain_valset(writer, loss_item, iter_count):

    writer.add_scalar('loss/val', loss_item, iter_count)

    print('| Validation | Iter - {} | Loss - {:.4f}  | '.format(iter_count, loss_item))
    
    return writer

def tensorboard_log_train_valset(writer, loss_item, iter_count):

    writer.add_scalar('loss/val', loss_item, iter_count)
    
    return writer


def tensorboard_log_val_classification(writer, accuracy_val_epoch, balanced_accuracy_val_epoch, loss_item, iter_count):
    
    writer.add_scalar('loss/val',loss_item , iter_count)
    writer.add_scalar('accuracy/val',accuracy_val_epoch, iter_count)
    writer.add_scalar('balanced_accuracy/val',balanced_accuracy_val_epoch, iter_count)
    
    print('| Validation | Iteration - {} | Loss - {:.4f} | accuracy - {:.4f} | balanced accuracy - {:.4f} |'.format(iter_count, loss_item, accuracy_val_epoch, balanced_accuracy_val_epoch))
    

def tensorboard_log_val_regression(config, writer, mae_val_epoch, r2_val_epoch, correlation_val, loss_item, iter_count,
                                   preds_val, targets_val, folder_to_save_model):
    
    writer.add_scalar('loss/val',loss_item , iter_count)
    writer.add_scalar('mae/val',mae_val_epoch, iter_count)
    writer.add_scalar('r2/val',r2_val_epoch, iter_count)
    writer.add_scalar('correlation/val',correlation_val, iter_count)

    if config['data']['dataset'] == 'HCP':
        plot_regression_results_HCP(np.concatenate(preds_val), np.concatenate(targets_val), os.path.join(folder_to_save_model,'results_val'), iter_count)
        
    print('| Validation | Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | Corr - {:.4f} | R2 - {:.4f} |'.format(iter_count, loss_item, mae_val_epoch, correlation_val, r2_val_epoch))



def log_classification(config, writer,optimizer, scheduler, preds, targets, loss_train_epoch, iteration ):
    
    accuracy_epoch = accuracy_score(preds,targets)
    balanced_accuracy_epoch = balanced_accuracy_score(preds,targets)

    writer.add_scalar('accuracy/train',accuracy_epoch, iteration)
    writer.add_scalar('balanced_accuracy/train',balanced_accuracy_epoch, iteration)

    if config['optimisation']['use_scheduler']:
        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            if config['optimisation']['warmup']:
                if (iteration+1)<config['optimisation']['nbr_step_warmup']:
                    print('| Iteration - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, round(accuracy_epoch,4), scheduler.get_lr()[0] ))
                else:
                    print('| Iteration - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, round(accuracy_epoch,4),optimizer.param_groups[0]['lr'] ))
            else:
                print('| Iteration - {} | Loss - {:.4f} | accuracy - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, round(accuracy_epoch,4),optimizer.param_groups[0]['lr'] ))
        else:
            print('| Iteration - {} | Loss - {:.4f} | accuracy - {:.2f} | balanced accuracy - {:.2f}% | LR - {}'.format(iteration+1, loss_train_epoch, accuracy_epoch*100, balanced_accuracy_epoch*100,scheduler.get_last_lr()[0] ))
    else:
        print('| Iteration - {} | Loss - {:.4f} | accuracy - {:.2f}% | balanced accuracy - {:.2f}% | LR - {}'.format(iteration+1, loss_train_epoch, accuracy_epoch*100, balanced_accuracy_epoch*100 ,optimizer.param_groups[0]['lr']))

    return writer


def log_regression(config, writer, optimizer, scheduler, preds, targets,loss_train_epoch, iteration, folder_to_save_model ):
    
    mae_epoch = np.mean(np.abs(np.concatenate(targets) - np.concatenate(preds)))
    r2_epoch = r2_score(np.concatenate(targets) , np.concatenate(preds))
    correlation = pearsonr(np.concatenate(targets).reshape(-1),np.concatenate(preds).reshape(-1))[0]

    writer.add_scalar('mae/train',mae_epoch, iteration+1)
    writer.add_scalar('r2/train',r2_epoch, iteration+1)
    writer.add_scalar('correlation/train',correlation, iteration+1)

    if config['data']['dataset'] == 'UKB':
        plot_regression_results_UKB(np.concatenate(preds), np.concatenate(targets), os.path.join(folder_to_save_model,'results_train'), iteration+1)
    elif config['data']['dataset'] == 'HCP':
        plot_regression_results_HCP(np.concatenate(preds), np.concatenate(targets), os.path.join(folder_to_save_model,'results_train'), iteration+1)

    if config['optimisation']['use_scheduler']:
        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            if config['optimisation']['warmup']:
                if (iteration+1)<config['optimisation']['nbr_step_warmup']:
                    print('| Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(iteration+1,loss_train_epoch, round(mae_epoch,4), scheduler.get_lr()[0] ))
                else:
                    print('| Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
            else:
                print('| Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, round(mae_epoch,4),optimizer.param_groups[0]['lr'] ))
        else:
            print('| Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | R2 - {:.4f} | Corr - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, mae_epoch, r2_epoch, correlation,  scheduler.get_last_lr()[0] ))
    else:
        print('| Iteration - {} | Loss - {:.4f} | MAE - {:.4f} | R2 - {:.4f} | Corr - {:.4f} | LR - {}'.format(iteration+1, loss_train_epoch, mae_epoch, r2_epoch, correlation,  optimizer.param_groups[0]['lr']))
    
    return writer

def log_pretrain(config,optimizer, scheduler,iter_count,loss_item ):
    

    if config['optimisation']['use_scheduler']:
        print('| Iter - {} | Loss - {:.4f} | LR - {}'.format( iter_count, loss_item, scheduler.get_last_lr()[0] ))
    else:
        print('| Iter - {} | Loss - {:.4f} | LR - {}'.format( iter_count, loss_item, optimizer.param_groups[0]['lr']))



def saving_ckpt_classification(config, best_accuracy, best_loss, iteration, balanced_accuracy_val_epoch,folder_to_save_model,
                               model, optimizer, loss_train_epoch):
    
    
    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_val_accuracy'] = float(best_accuracy)
    config['results']['best_val_loss'] = float(best_loss)
    config['results']['best_val_balanced_accuracy'] = float(balanced_accuracy_val_epoch)
    config['results']['best_val_epoch'] = iteration+1
    config['results']['training_finished'] = False

    ###############################
    ######    SAVING CKPT    ######
    ###############################

    if config['training']['save_ckpt']:
        torch.save({'epoch': iteration+1,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train_epoch,
                    },
                    os.path.join(folder_to_save_model,'checkpoint_best.pth'))



def saving_ckpt_regression(config, best_mae, best_loss, iteration,r2_val_epoch, folder_to_save_model,
                           model, optimizer,loss_train_epoch):
    

    config['logging']['folder_model_saved'] = folder_to_save_model
    config['results'] = {}
    config['results']['best_val_mae'] = float(best_mae)
    config['results']['best_val_loss'] = float(best_loss)
    config['results']['best_val_r2'] = float(r2_val_epoch)
    config['results']['best_val_epoch'] = iteration
    config['results']['training_finished'] = False 

    ###############################
    ######    SAVING CKPT    ######
    ###############################

    if config['training']['save_ckpt']:
        torch.save({'epoch': iteration,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train_epoch,
                    },
                    os.path.join(folder_to_save_model,'checkpoint_best.pth'))
        

def save_reconstruction_pretain(config, output_batch, input_batch, inputs, masked_indices, unmasked_indices,iter_count,folder_to_save_model):

    #mesh_resolution
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    if config['MODEL'] == 'sit':    
        channels = config['transformer']['channels']
    num_channels = len(channels)

    if (config['SSL'] == 'mae' and config['pretraining_mae']['save_reconstruction']) or (config['SSL'] == 'smae' and config['pretraining_smae']['save_reconstruction'])or (config['SSL'] == 'vsmae' and config['pretraining_smae']['save_reconstruction']) :
        #print('saving reconstruction')
        save_reconstruction_mae(output_batch,
                                input_batch,
                                inputs, 
                                num_patches,
                                num_vertices,
                                ico_grid,
                                num_channels,
                                masked_indices[:1],
                                unmasked_indices[:1],
                                str(int(iter_count)).zfill(6),
                                folder_to_save_model,
                                split='train',
                                path_to_workdir=config['data']['path_to_workdir'],
                                id='0',
                                server = config['SERVER']
                                )



def save_reconstruction_pretrain_fmri(config,
                                      output_batch,
                                      input_batch,
                                      inputs,
                                      masked_indices,
                                      unmasked_indices,
                                      iter_count,
                                      folder_to_save_model,):

    #mesh_resolution
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    nbr_frames = config['fMRI']['nbr_frames'] 

    if (config['SSL'] == 'vsmae' and config['pretraining_smae']['save_reconstruction']) :
        #print('saving reconstruction')
        save_reconstruction_mae_fmri(output_batch,
                                    input_batch,
                                    inputs, 
                                    num_patches,
                                    ico_grid,
                                    nbr_frames,
                                    masked_indices,
                                    unmasked_indices,
                                    str(int(iter_count)).zfill(6),
                                    folder_to_save_model,
                                    split='train',
                                    path_to_workdir=config['data']['path_to_workdir'],
                                    id='0',
                                    server = config['SERVER'],
                                    masking_type = config['pretraining_vsmae']['masking_type']
                                    )




def save_reconstruction_pretrain_fmri_valset(config,
                                      output_batch,
                                      input_batch,
                                      inputs,
                                      masked_indices,
                                      unmasked_indices,
                                      iter_count,
                                      folder_to_save_model,
                                      id,
                                      ):

    #mesh_resolution
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    nbr_frames = config['fMRI']['nbr_frames'] 

    if (config['SSL'] == 'vsmae' and config['pretraining_smae']['save_reconstruction']) :
        #print('saving reconstruction')
        save_reconstruction_mae_fmri(output_batch,
                                    input_batch,
                                    inputs, 
                                    num_patches,
                                    ico_grid,
                                    nbr_frames,
                                    masked_indices,
                                    unmasked_indices,
                                    str(int(iter_count)).zfill(6),
                                    folder_to_save_model,
                                    split='val',
                                    path_to_workdir=config['data']['path_to_workdir'],
                                    id=id,
                                    server = config['SERVER'],
                                    masking_type = config['pretraining_vsmae']['masking_type']
                                    )



def saving_ckpt_pretrain(config, iter_count, best_loss_it, best_loss_val,folder_to_save_model,
                               model, ssl, optimizer):
    
    config['results'] = {}
    config['results']['best_iter'] = iter_count
    config['results']['best_current_loss'] = best_loss_it
    config['results']['best_current_loss_validation'] = best_loss_val

    with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


    torch.save({ 'epoch':iter_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':best_loss_it,
                },
                os.path.join(folder_to_save_model, 'encoder-best.pt'))
    torch.save({ 'epoch':iter_count,
                'model_state_dict': ssl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':best_loss_it,
                },
                os.path.join(folder_to_save_model, 'encoder-decoder-best.pt'))

        
    return config


