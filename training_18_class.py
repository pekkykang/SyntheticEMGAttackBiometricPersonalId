import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import model_18
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

subject_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]  # 'please rewrite the data process part according to your data naming logic'
gesture_list = [1, 2, 3, 4, 5, 6, 7, 8] # 'please rewrite the data process part according to your data naming logic'
trial_list = [1,  3,  5, 7,  9] # 'please rewrite the data process part according to your data naming logic'
trial_list_test = [2, 4, 6, 8, 10] # 'please rewrite the data process part according to your data naming logic'
path_n = 'data path' # 'please rewrite the data process part according to your data naming logic'
labels_name = ["1", "2", "3", "4", "5", "6", "7", "8", '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'] # 'please rewrite the data process part according to your data naming logic'

num_classes = len(subject_list)

emg_num = 128
Hz = 1000
n_steps = Hz*0.128  
n_update = 0.5 * n_steps

batch_size = 36
epochs = 200
patience = 0.2*epochs
#decay_epoch = 0.1*epochs

C_list = np.zeros((num_classes, num_classes))  
acc_list = []
recall_list = []
f1_list = []


for gesture in gesture_list:
    # data process
    # please rewrite the data process part according to your data naming logic
    # you need to get n*128*128 data segments
    
    data_subject = []
    information_list = np.zeros([0, 2])
    data_subject_test = []
    information_list_test = np.zeros([0, 2])
    for subject in subject_list:
        for trial in trial_list:

            data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)

            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

            for k in data_segmented:
                data_subject.append(k)
                information = np.array([[subject-1, gesture]])
                information_list = np.concatenate((information_list, information), axis=0)
        for trial in trial_list_test:
            data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)

            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)

            for k in data_segmented:
                data_subject_test.append(k)
                information = np.array([[subject-1, gesture]])
                information_list_test = np.concatenate((information_list_test, information), axis=0)
    data_subject = np.array(data_subject)
    label_subject = np.array(information_list[:, 0])
    data_subject_test = np.array(data_subject_test)
    label_subject_test = np.array(information_list_test[:, 0])

    train_x, val_x, train_y, val_y = train_test_split(data_subject, label_subject, test_size=0.1, random_state=1)
    train_x = torch.from_numpy(train_x).to(torch.float32).cuda()
    train_y = torch.from_numpy(train_y).to(torch.float32).cuda()

    torch_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)

    val_x = torch.from_numpy(val_x).to(torch.float32).cuda()
    val_y = torch.from_numpy(val_y).to(torch.float32).cuda()

    test_x = torch.from_numpy(data_subject_test).to(torch.float32).cuda()
    test_y = torch.from_numpy(label_subject_test).to(torch.float32).cuda()


    model = model_18.Net_18()
    print("Model initialized")
    model.cuda()  
    torch.save(model, 'please use your model path')
    min_valid_loss = 120
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimize all cnn parameters 
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    curr_patience = patience

    # Learning rate update schedulers  
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #    optimizer, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step
    #)
    #with torch.no_grad():  

    for e in range(epochs):
        model.train()  
        avg_train_loss = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            b_y = b_y.cuda()

            optimizer.zero_grad()  # clear gradients for this training step

            output = model(b_x)  # cnn output
            
            loss = loss_func(output, b_y.long())

            loss.requires_grad_(True)

            loss.backward()
            optimizer.step()  # apply gradients
            avg_loss = loss.item()
            avg_train_loss += avg_loss
        avg_train_loss = avg_train_loss
        print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        model.eval() 
        val_output = model(val_x)
        valid_loss = loss_func(val_output, val_y.long())
        avg_valid_loss = valid_loss.item()
        pred_y = torch.max(val_output.cpu(), 1)[1]

        accuracy = accuracy_score(val_y.cpu(), pred_y)
        print('Validation accuracy: %.2f' % accuracy)
        avg_valid_loss = avg_valid_loss #/ len(val_y)
        print("Validation loss is: {}".format(avg_valid_loss))

        if (avg_valid_loss < min_valid_loss):
            curr_patience = patience 
            min_valid_loss = avg_valid_loss
            torch.save(model, 'please use your model path')
            print("Found new best model, save to disk")
        else:
            curr_patience -= 1
        if curr_patience <= 0:
            break
        #lr_scheduler.step() 

    """
    with torch.no_grad():
        
        best_model = torch.load('please use your model path')
        best_model.eval()

        test_output = best_model(test_x)

        pred_y = torch.max(test_output.cpu(), 1)[1]
        accuracy = accuracy_score(test_y.cpu(), pred_y)
        recall = recall_score(test_y.cpu(), pred_y, average='macro')
        f1 = f1_score(test_y.cpu(), pred_y, average='macro')
        C = confusion_matrix(test_y.cpu(), pred_y)

        plot_confusion_matrix_bigger(C, labels_name, "confusion matrix " + str(gesture) + " accuracy " + str('%.4f' %accuracy))
        plt.show()
        plt.close()

        print('Gesture %s' % gesture)
        print('Test accuracy: %.2f' % accuracy)
        print('Test recall: %.2f' % recall)
        print('Test f1: %.2f' % f1)

        acc_list.append(accuracy)
        recall_list.append(recall)
        f1_list.append(f1)
        C_list = C_list + C
"""
