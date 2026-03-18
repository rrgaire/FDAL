from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import models.resnet as resnet
import torch.nn.functional as F

##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def test(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):


    models['backbone'].train()
    if method == 'lloss' or method == 'TA-VAAL':
        models['module'].train()
    elif method == 'FDAL':
        models['decoder'].train()
        models['sampler'].train()
        models['teacher'].eval()
        unlabeled_data = iter(dataloaders['unlabeled'])

    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].zero_grad()
        elif method == 'FDAL':
            optimizers['decoder'].zero_grad()
            optimizers['sampler'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        target_loss = criterion['ce_loss'](scores, labels)

        if method == 'lloss' or method == 'TA-VAAL':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()
        
        elif method == 'FDAL':

            try:
                unlab_inputs, _, _ = next(unlabeled_data)
            except StopIteration:
                unlabeled_data = iter(dataloaders['unlabeled'])
                unlab_inputs, _, _ = next(unlabeled_data)
            unlab_inputs = unlab_inputs.cuda()

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)  

            
            loss = m_backbone_loss
            
            with torch.no_grad():
                teach_lab_feats, teach_lab_logits = models['teacher'](inputs)

            dec_lab_feats = models['decoder'](features[4])
            
            feat_loss = criterion['mse_loss'](dec_lab_feats, teach_lab_feats)
            logit_loss = criterion['mse_loss'](scores, teach_lab_logits)

            loss += (FEAT_WT * feat_loss + LOGIT_WT * logit_loss)

            if epoch >= EPOCH_FDAL:

                disc_lab = models['sampler'](dec_lab_feats).squeeze(1)

                _, _, unlab_feats = models['backbone'](unlab_inputs) 
                dec_unlab_feats = models['decoder'](unlab_feats[4])
                disc_unlab = models['sampler'](dec_unlab_feats).squeeze(1)

                lab_real_preds = torch.ones(inputs.size(0)).cuda()
                unlab_real_preds = torch.ones(unlab_inputs.size(0)).cuda()

                disc_loss = ( criterion['bce_loss'](disc_lab, lab_real_preds) + criterion['bce_loss'](disc_unlab, unlab_real_preds) ) / 2
            
                loss += ADV_WT * disc_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['decoder'].step()

            if epoch >= EPOCH_FDAL:
                optimizers['sampler'].zero_grad()

                with torch.no_grad():
                    _, _, lab_feats = models['backbone'](inputs) 
                    lb_feat = models['decoder'](lab_feats[4])

                    _, _, unlab_feats = models['backbone'](unlab_inputs) 
                    unlb_feat = models['decoder'](unlab_feats[4])
                
                labeled_preds = models['sampler'](lb_feat).squeeze(1)
                unlabeled_preds = models['sampler'](unlb_feat).squeeze(1)
                
                lab_real_preds = torch.ones(inputs.size(0))
                unlab_fake_preds = torch.zeros(unlab_inputs.size(0))

                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()

                disc_loss = ( criterion['bce_loss'](labeled_preds, lab_real_preds) + criterion['bce_loss'](unlabeled_preds, unlab_real_preds) ) / 2

                disc_loss.backward()
                optimizers['sampler'].step()
        
            
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss = m_backbone_loss

            loss.backward()
            optimizers['backbone'].step()


            
    return loss

def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    best_acc = 0.
    num_epochs = EPOCH

    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            schedulers['module'].step()
        if method == 'FDAL' and epoch >= EPOCH_FDAL:
            schedulers['decoder'].step()
            schedulers['sampler'].step()

        if epoch % 10  == 0:
            acc = test(models, epoch, method, dataloaders, mode='test')
            if best_acc < acc:
                best_acc = acc
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    
    print('>> Finished.')
    return best_acc