
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from cycleGAN_dependence import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from Tool_Pretreatment import *
from sklearn.model_selection import train_test_split
from Tool_Pretreatment import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from Tool_Visualization import *
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from Tool_init import *
import numpy as np
import argparse

import pandas as pd
import csv
import codecs
from Tool_HEMG_visualization import *

def segment_2D(n_width, n_update, input_data, num_axis):
    data = []
    n_width = int(n_width)
    n_update = int(n_update)
    data_len = input_data.shape[0]
    segment_num = int(np.floor(data_len/n_update)-(n_width/n_update)+1)
    if n_width/n_update == 1:
        segment_num = int(np.floor(data_len / n_update))
    for i_win in range(segment_num):
        temp = input_data[i_win*n_update:i_win*n_update+n_width, 0:num_axis]
        data.append(temp)
    data = np.array(data)
    return data


def select_data(subject, gesture, trial, axis, path):
    file_name = '%d' % subject + '-' + '%d' % gesture + '-' + '%d' % trial + '.csv'
    file_path = path + '//' + file_name
    all_axis = ['ALL']
    for i in range(0, len(all_axis)):
        if axis == all_axis[i]:
            if axis =='ALL':
                select_cols = range(0, 128)
                data = pd.read_csv(file_path, header=None, usecols=select_cols)
                data = np.array(data)
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=14, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="subject2subject", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=8, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()

input_shape = (opt.channels, opt.img_height, opt.img_width)

path_n = 'data path'

emg_num = 128
Hz = 1000
n_steps = Hz*0.128
n_update = 0.5 * n_steps

gesture_list = 'please rewrite the data process part according to your data naming logic'
subject_a_list = 'please rewrite the data process part according to your data naming logic'
subject_b_list = 'please rewrite the data process part according to your data naming logic'
trial_list = 'please rewrite the data process part according to your data naming logic'
trial_list_b ='please rewrite the data process part according to your data naming logic'
trial_list_test = 'please rewrite the data process part according to your data naming logic'



data_subject = []
for gesture in gesture_list:
    # data process
    # please rewrite the data process part according to your data naming logic
    # you need to get n*128*128 data segments
    for index in range(0, len(subject_a_list)):
        writer_a = np.zeros([0, 129])
        writer_b = np.zeros([0, 129])
        subject_a = subject_a_list[index]
        data_a = []
        for trial in trial_list:
            data_trial = select_data(subject_a, gesture, trial, 'ALL', path=path_n)
            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)
            for k in data_segmented:
                data_a.append(k)
        data_a = np.array(data_a)
        subject_b = subject_b_list[index]
        data_b = []
        information_list_b = np.zeros([0, 2])
        for trial in trial_list_b:
            data_trial = select_data(subject_b, gesture, trial, 'ALL', path=path_n)
            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)
            for k in data_segmented:
                data_b.append(k)
        data_b = np.array(data_b)

        data_a_train, _, data_b_train, _ = train_test_split(data_a, data_b, test_size=0.1,
                                                                                random_state=1)

        torch_dataset = Data.TensorDataset(torch.tensor(data_a_train), torch.tensor(data_b_train))
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=opt.batch_size, shuffle=False)


        data_a_test = []
        for trial in trial_list_test:

            data_trial = select_data(subject_a, gesture, trial, 'ALL', path=path_n)

            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

            for k in data_segmented:
                data_a_test.append(k)
        data_a_test = np.array(data_a_test)

        data_b_test = []
        information_list_b = np.zeros([0, 2])
        for trial in trial_list_test:

            data_trial = select_data(subject_b, gesture, trial, 'ALL', path=path_n)

            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

            for k in data_segmented:
                data_b_test.append(k)
        data_b_test = np.array(data_b_test)

        torch_dataset = Data.TensorDataset(torch.tensor(data_a_test), torch.tensor(data_b_test))
        val_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1, shuffle=False)


        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        criterion_GAN = torch.nn.MSELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()
        # Initialize generator and discriminator
        G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
        G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
        D_A = Discriminator(input_shape)
        D_B = Discriminator(input_shape)
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
        optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )

        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        prev_time = time.time()
        for epoch in range(opt.epoch, opt.n_epochs):
            for i, (d_a, d_b) in enumerate(train_loader):
                # Set model input
                real_A = Variable(d_a.type(Tensor))
                real_B = Variable(d_b.type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))),
                                 requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))),
                                requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                G_AB.train()
                G_BA.train()

                optimizer_G.zero_grad()

                x = G_BA(real_A)
                y = G_AB(real_B)
                loss_id_A = criterion_identity(x, real_A)
                loss_id_B = criterion_identity(y, real_B)

                loss_identity = (loss_id_A + loss_id_B) / 2

                fake_B = G_AB(real_A)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # loss：Cycle loss
                recov_A = G_BA(fake_B)
                loss_cycle_A = criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)

                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

                loss_G.backward()
                optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                optimizer_D_A.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                optimizer_D_B.zero_grad()

                # Real loss
                loss_real = criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

                # Determine approximate time left
                batches_done = epoch * len(train_loader) + i
                batches_left = opt.n_epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(train_loader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                    )
                )

                # If at sample interval save image

                if batches_done % opt.sample_interval == 0:

                    input_data_a, input_data_b = next(iter(val_loader))
                    """Saves a generated sample from the test set"""
                    G_AB.eval()
                    G_BA.eval()
                    real_A = Variable(input_data_a.type(Tensor))
                    fake_B = G_AB(real_A)
                    real_B = Variable(input_data_b.type(Tensor))
                    fake_A = G_BA(real_B)
                    # Arange images along x-axis
                    real_A = make_grid(real_A, nrow=5, normalize=True)
                    real_A = real_A.cpu().detach().numpy()[0]
                    real_B = make_grid(real_B, nrow=5, normalize=True)
                    real_B = real_B.cpu().detach().numpy()[0]
                    fake_A = make_grid(fake_A, nrow=5, normalize=True)
                    fake_A = fake_A.cpu().detach().numpy()[0]
                    fake_B = make_grid(fake_B, nrow=5, normalize=True)
                    fake_B = fake_B.cpu().detach().numpy()[0]

                    data_to_img = np.concatenate((real_A, fake_A, real_B, fake_B), axis=1)

                    if loss_D.item() < 0.15:

                        generated_a = []
                        generated_b = []
                        for k, (K_a, K_b) in enumerate(val_loader):
                            if k < 5:
                                real_A = Variable(K_a.type(Tensor))
                                real_B = Variable(K_b.type(Tensor))
                                fake_B = G_AB(real_A).cpu().detach().numpy()[0]
                                fake_A = G_BA(real_B).cpu().detach().numpy()[0]
                                generated_a.append(fake_A)
                                generated_b.append(fake_B)
                        generated_a = np.array(generated_a)
                        generated_b = np.array(generated_b)

                        for w in generated_a:

                            batch_label = batches_done * np.ones((len(w), 1))
                            w = np.concatenate((w, batch_label), axis=1)
                            writer_a = np.concatenate((writer_a, w), axis=0)
                        for w in generated_b:
                            batch_label = batches_done * np.ones((len(w), 1))
                            w = np.concatenate((w, batch_label), axis=1)
                            writer_b = np.concatenate((writer_b, w), axis=0)

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        writer_a = pd.DataFrame(data=writer_a)
        writer_a.to_csv('path to save data', index=False, header=False)
        writer_b = pd.DataFrame(data=writer_b)
        writer_b.to_csv('path to save data', index=False, header=False)

