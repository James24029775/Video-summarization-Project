import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
from model_compat import DSN
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
# import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
import pandas as pd
from sklearn.manifold import TSNE


def test(epoch, loader):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 28

    ###################
    # load data       #
    ###################

    # img_transform_mnist = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5), std=(0.5))
    # ])

    # img_transform_mnist_m = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    model_root = 'model'

    ####################
    # load model       #
    ####################

    # my_net = DSN(code_size=100, n_class=2)
    # checkpoint = torch.load(os.path.join(model_root, 'dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth'))
    # torch.load(FE, os.path.join('trained_weight', str(epoch)+'_FE.pth'))
    my_net = torch.load(os.path.join('trained_weight', str(epoch)+'_DSN.pth'))
    my_net.eval()

    FE = torch.load(os.path.join('trained_weight', str(epoch)+'_FE.pth'))
    FE.eval()

    if cuda:
        my_net = my_net.cuda()
        FE = FE.cuda()

    S_loader = loader[0]
    B_loader = loader[1]

    len_dataloader = len(S_loader)
    data_iter = iter(S_loader)

    i = 0
    n_total = 0
    n_correct = 0

    total_S_labels = []
    total_S_preds = []
    first = True

    while i < len_dataloader:

        data_input = data_iter.next()
        img, label = data_input

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        inputv_img = FE(inputv_img)

        result, feature = my_net(input_data=inputv_img, mode='source', rec_scheme='share')
        pred = result[3].data.max(1, keepdim=True)[1]


        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

        #! mine
        total_S_labels.extend(classv_label.data.cpu().numpy())
        total_S_preds.extend(pred.data.cpu().numpy())
        if first:
            S_features = feature.data.cpu().numpy()
            first = False
        else:
            S_features = np.vstack((S_features, feature.data.cpu().numpy()))

    accu = n_correct * 1.0 / n_total
    S_F1 = f1_score(total_S_labels, total_S_preds)

    print ('epoch: %d, accuracy and F1 of the source dataset: %f, %f' % (epoch, accu, S_F1))
    
    


    len_dataloader = len(B_loader)
    data_iter = iter(B_loader)

    i = 0
    n_total = 0
    n_correct = 0

    total_B_labels = []
    total_B_preds = []
    first = True

    while i < len_dataloader:

        data_input = data_iter.next()
        img, label = data_input

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        inputv_img = FE(inputv_img)

        result, feature = my_net(input_data=inputv_img, mode='source', rec_scheme='share')
        pred = result[3].data.max(1, keepdim=True)[1]


        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

        #! mine
        total_B_labels.extend(classv_label.data.cpu().numpy())
        total_B_preds.extend(pred.data.cpu().numpy())
        if first:
            B_features = feature.data.cpu().numpy()
            first = False
        else:
            B_features = np.vstack((B_features, feature.data.cpu().numpy()))

    accu = n_correct * 1.0 / n_total
    B_F1 = f1_score(total_B_labels, total_B_preds)

    print ('epoch: %d, accuracy and F1 of the target dataset: %f, %f' % (epoch, accu, B_F1))


    # plot GT B
    total_B_labels = np.array(total_B_labels)
    total_B_preds = np.array(total_B_preds)
    B_df = pd.DataFrame(B_features)
    total_B_labels = pd.DataFrame(total_B_labels, columns = ['label'])
    total_B_preds = pd.DataFrame(total_B_preds, columns = ['preds'])
    tsne = TSNE(random_state=17)
    tsne_repr = tsne.fit_transform(B_df)
    plt.clf()
    plt.scatter(
        tsne_repr[:, 0],
        tsne_repr[:, 1],
        c=total_B_labels['label'].map({0: "magenta", 1: "blue"}),
        alpha=0.5,
    )
    plt.savefig(os.path.join('png', 'label', str(epoch)))

    # plot pred B
    tsne = TSNE(random_state=17)
    tsne_repr = tsne.fit_transform(B_df)
    plt.clf()
    plt.scatter(
        tsne_repr[:, 0],
        tsne_repr[:, 1],
        c=total_B_preds['preds'].map({0: "magenta", 1: "blue"}),
        alpha=0.5,
    )
    plt.savefig(os.path.join('png', 'pred', str(epoch)))

    # plot similarity
    total_S_labels = np.array(total_S_labels)
    total_S_preds = np.array(total_S_preds)
    S_df = pd.DataFrame(S_features)
    total_df = pd.concat([S_df, B_df],axis=0)
    domain_label = np.zeros(len(total_df))
    domain_label[:len(S_df)] = 0
    domain_label[len(S_df):] = 1
    domain_label = pd.DataFrame(domain_label, columns = ['label'])

    tsne = TSNE(random_state=17)
    tsne_repr = tsne.fit_transform(total_df)
    plt.clf()
    plt.scatter(
        tsne_repr[:, 0],
        tsne_repr[:, 1],
        c=domain_label['label'].map({0: "red", 1: "green"}),
        alpha=0.5,
    )
    plt.savefig(os.path.join('png', 'sim', str(epoch)))