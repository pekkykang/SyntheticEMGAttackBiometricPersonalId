import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import model_144class
import pandas as pd


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

subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
gesture_list = [1, 2, 3, 4, 5, 6, 7, 8]

trial_all_list = [1,2,3,4,5,6,7,8,9,10]
path_n = './H-EMG-N/'  # normalized data
labels_name = [str(i) for i in range(1, 145)]
num_classes = int(8*18)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
emg_num = 128
Hz = 1000
n_steps = Hz * 0.128
n_update = 0.5 * n_steps

batch_size = 512
epochs = 800
patience = 100
# decay_epoch = 0.1*epochs

C_list = np.zeros((num_classes, num_classes))
acc_list = []
recall_list = []
f1_list = []

for trial in trial_all_list:
    label_current = 0
    data_subject = []
    information_list = np.zeros([0, 2])
    data_subject_test = []
    information_list_test = np.zeros([0, 2])
    for gesture in gesture_list:
        for subject in subject_list:

            for trial_current in trial_all_list:
                if trial_current == trial:
                    data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)

                    data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

                    for k in data_segmented:
                        data_subject_test.append(k)
                        information = np.array([[label_current, gesture]])
                        information_list_test = np.concatenate((information_list_test, information), axis=0)
                else:
                    data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)

                    data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

                    for k in data_segmented:
                        data_subject.append(k)
                        information = np.array([[label_current, gesture]])
                        information_list = np.concatenate((information_list, information), axis=0)

            label_current = label_current + 1

    data_subject = np.array(data_subject)
    label_subject = np.array(information_list[:, 0])
    data_subject_test = np.array(data_subject_test)
    label_subject_test = np.array(information_list_test[:, 0])


    train_x, val_x, train_y, val_y = train_test_split(data_subject, label_subject, test_size=0.2, random_state=1)
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y)

    torch_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)

    val_y = torch.from_numpy(val_y)
    torch_dataset = Data.TensorDataset(val_x, val_y)
    val_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)

    test_x = torch.from_numpy(data_subject_test).float()
    test_y = torch.from_numpy(label_subject_test)
    torch_dataset = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True)


    model = model_144class.Net_144().cuda().train()
    print("Model initialized")
    torch.save(model, './saved_models/144/Net_144'+str(trial)+'.pt')
    min_valid_loss = 6000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    curr_patience = patience

    for e in range(epochs):
        model.train()
        avg_train_loss = 0.0
        avg_valid_loss = 0.0
        avg_valid_acc = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader



            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()  # clear gradients for this training step

            output = model(b_x)  # cnn output

            loss = loss_func(output, b_y.long()) / len(train_loader)

            loss.backward()
            optimizer.step()  # apply gradients

            avg_train_loss += loss.item()
        avg_train_loss = avg_train_loss
        print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))
        del b_x
        del b_y
        for step, (val_x, val_y) in enumerate(val_loader):

            model.eval()
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_output = model(val_x)

            valid_loss = loss_func(val_output, val_y.long()) / len(val_loader)
            avg_valid_loss += valid_loss.item()

            pred_y = torch.max(val_output.cpu(), 1)[1]

            acc = accuracy_score(val_y.cpu(), pred_y)

            avg_valid_acc = avg_valid_acc+acc

        avg_valid_loss = avg_valid_loss
        print("Validation loss is: {}".format(avg_valid_loss))
        avg_valid_acc = avg_valid_acc / len(val_loader)
        print('Validation accuracy: %.2f' % avg_valid_acc)
        del val_x
        del val_y
        if (avg_valid_loss < min_valid_loss):
            curr_patience = patience
            min_valid_loss = avg_valid_loss
            torch.save(model, './saved_models/144/Net_144'+str(trial)+'.pt')
            print("Found new best model, save to disk")
        else:
            curr_patience -= 1
        if curr_patience <= 0:
            break
    
    with torch.no_grad():
        best_model = torch.load('./saved_models/144/Net_144'+str(trial)+'.pt')
        best_model.eval().cpu()
        for step, (test_x, test_y) in enumerate(test_loader):
            test_output = best_model(test_x)
            pred_y = torch.max(test_output.cpu(), 1)[1]
            accuracy = accuracy_score(test_y.cpu(), pred_y)
            recall = recall_score(test_y.cpu(), pred_y, average='macro')
            f1 = f1_score(test_y.cpu(), pred_y, average='macro')

            print('Test accuracy: %.2f' % accuracy)
            print('Test recall: %.2f' % recall)
            print('Test f1: %.2f' % f1)
            break
        acc_list.append(accuracy)
        recall_list.append(recall)
        f1_list.append(f1)
    torch.cuda.empty_cache()

print(acc_list)
print(recall_list)
print(f1_list)

