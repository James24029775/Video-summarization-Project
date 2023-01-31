import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE
from test import test
from data.prepare_data import generate_dataloader
from src.models import Wav2Vec2ForSpeechClassification
import torch
import torch.nn as nn
from transformers import AutoConfig
from model_2 import wav2vec2

######################
# params             #
######################

model_root = 'model'
cuda = True
cudnn.benchmark = True
lr = 2e-3
image_size = 28
n_epoch = 1000
step_decay_weight = 0.95
weight_decay = 1e-6
# DSN�@�s�b�T��loss functions�A���O�ѥH�U�T���v���i�汱��
alpha_weight = 1
beta_weight = 0.1
gamma_weight = 0.25

lr_decay_step = 2000
active_domain_loss_step = 1100
momentum = 0.9
worker = 8

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################
# load data           #
#######################

class args:
    num_classes_s = 2
    lr = 2e-2
    da = True
    batch_size = 8
    weight_decay = 1e-4

source_train_loader, target_train_loader, source_val_loader, target_val_loader = generate_dataloader(args)

#####################
#  load model       #
#####################

# ���Jwav2vec2.0�һݪ��]�w�ɮסB�w�V�m�v��
class model_args:
    config_name = "pretrained/config.json"
    model_name_or_path = "pretrained/pytorch_model.bin"
    cache_dir = None
    model_revision = None
    use_auth_token = None

# ��l�ƨ��AI�ҫ��A���O�Owav2vec2.0�]�N�Obackbone�A�H��DSN
my_net = DSN(code_size=100, n_class=2)
FE = wav2vec2(model_args, args)

# ��Ӧ��s�U�V�m�n���v�������A�i�ϥΥH�U�����J�v��
# my_net = torch.load(os.path.join('trained_weight', 'best_140_DSN.pth'))
# FE = torch.load(os.path.join('trained_weight', 'best_140_FE.pth'))

#####################
# setup optimizer   #
#####################


def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print ('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
FE_opt = optim.SGD(FE.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.CrossEntropyLoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    FE = FE.cuda()
    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

for p in FE.parameters():
    FE.requires_grad = True

#############################
# training network          #
#############################

# �@�@�����dataloaders�n�Q���N�A���u�����u���@����ƶ]���A�]���j����פ]�u��]���u������
len_dataloader = min(len(source_train_loader), len(target_train_loader))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in range(n_epoch):

    data_source_iter = iter(source_train_loader)
    data_target_iter = iter(target_train_loader)

    i = 0

    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = data_target_iter.next()
        # t_img�Mt_label���O�O���ۭ��T���e�H�μ��ҡA��ɻ{���ܼƦW�٨Sԣ�v�T�N����F��\_(?)_/��
        t_img, t_label = data_target

        # �N�ҫ�����k�s
        my_net.zero_grad()
        # FE.zero_grad()
        loss = 0
        batch_size = len(t_label)

        # ���κ�input_img����l�j�p���]�A�ϥ���ӷ|�Qresize���z�Q�j�p(�]�N�Ot_img���Ϊ�)
        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        target_inputv_img = Variable(input_img)
        target_classv_label = Variable(class_label)
        target_domainv_label = Variable(domain_label)

        
        # ��backbone���N���T�S�x��X
        target_inputv_img = FE(target_inputv_img)

        # �H�U���󬰤F�ݼҫ��yí�w�~�i�@�B�V�m�o��loss
        if current_step > active_domain_loss_step:
            # Ĳ�o����h�}�l�[�JDANN�H���hdomain gap
            p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            p = 2. / (1. + np.exp(-10 * p)) - 1

            # activate domain loss
            result, _ = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
            loss += target_dann
        else:
            target_dann = Variable(torch.zeros(1).float().cuda())
            result, _ = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
            target_privte_code, target_share_code, _, target_rec_code = result

        # �p��difference loss(���F��disentanglement)�H��reconstruction loss(���T�O�G��encoders��S�x���~��)
        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
        loss += target_simse

        loss.backward()
        optimizer.step()
        FE_opt.step()

        ##################################
        # source data training            #
        ##################################

        # Source dataset�]�O�@�˪��B�J
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        FE.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        source_inputv_img = Variable(input_img)
        source_classv_label = Variable(class_label)
        source_domainv_label = Variable(domain_label)

        source_inputv_img = FE(source_inputv_img)

        if current_step > active_domain_loss_step:
            # activate domain loss
            result, _ = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
            loss += source_dann
            
        else:
            source_dann = Variable(torch.zeros(1).float().cuda())
            result, _ = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

        # �ߤ@�t�����a��b��source dataset�����ҡA�]���i�H�i��ʷ����ǲ�
        source_classification = loss_classification(source_class_label, source_classv_label)
        loss += source_classification

        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff
        source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
        loss += source_simse

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        FE_opt = exp_lr_scheduler(optimizer=FE_opt, step=current_step)
        optimizer.step()
        FE_opt.step()

        i += 1
        current_step += 1
    print ('source_classification: %f, source_dann: %f, source_diff: %f, ' \
          'source_mse: %f, source_simse: %f, \ntarget_dann: %f, target_diff: %f, ' \
          'target_mse: %f, target_simse: %f' \
          % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
             source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
             target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))

    print ('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))
    

    if epoch % 10 == 0:
        torch.save(my_net, os.path.join('trained_weight', str(epoch)+'_DSN.pth'))
        torch.save(FE, os.path.join('trained_weight', str(epoch)+'_FE.pth'))
        test(epoch=epoch, loader=[source_val_loader, target_val_loader])

print ('done')





