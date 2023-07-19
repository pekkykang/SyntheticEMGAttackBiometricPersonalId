"""

针对每个手势计算个体的识别精度

"""

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

subject_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#subject_list = [1,3,2,4,6,8,11,13,10,12,17,18]
gesture_list = [1, 2, 3, 4, 5, 6, 7, 8]
trial_list = [1,  3,  5, 7,  9]
trial_list_test = [2, 4, 6, 8, 10]
path_n = './H-EMG-N/'
labels_name = ["1", "2", "3", "4", "5", "6", "7", "8", '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

num_classes = len(subject_list)

emg_num = 128
Hz = 1000
n_steps = Hz*0.128  # 窗长128ms
n_update = 0.5 * n_steps


"""模型的训练参数"""
batch_size = 36
epochs = 200
patience = 0.2*epochs
#decay_epoch = 0.1*epochs

C_list = np.zeros((num_classes, num_classes))  # 用于储存混淆矩阵
acc_list = []
recall_list = []
f1_list = []


for gesture in gesture_list:
    data_subject = []
    information_list = np.zeros([0, 2])
    data_subject_test = []
    information_list_test = np.zeros([0, 2])
    for subject in subject_list:
        # 构造A数据
        for trial in trial_list:
            # 将csv文件导入矩阵当中
            data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)
            # 将数据集进行归一化处理
            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)
            # 存储数据
            for k in data_segmented:
                data_subject.append(k)
                information = np.array([[subject-1, gesture]])  # pytorch不用热编码，标签从0开始 这里很重要经常报错
                information_list = np.concatenate((information_list, information), axis=0)
        for trial in trial_list_test:
            data_trial = select_data(subject, gesture, trial, 'ALL', path=path_n)
            # 将数据集进行归一化处理
            data_segmented = segment_2D(n_steps, n_update, data_trial, num_axis=emg_num)
            # 存储数据
            for k in data_segmented:
                data_subject_test.append(k)
                information = np.array([[subject-1, gesture]])  # pytorch不用热编码，标签从0开始 这里很重要经常报错
                information_list_test = np.concatenate((information_list_test, information), axis=0)
    data_subject = np.array(data_subject)
    label_subject = np.array(information_list[:, 0])
    data_subject_test = np.array(data_subject_test)
    label_subject_test = np.array(information_list_test[:, 0])


    # 把后两个作为测试集

    train_x, val_x, train_y, val_y = train_test_split(data_subject, label_subject, test_size=0.2, random_state=1)
    train_x = torch.from_numpy(train_x).to(torch.float32).cuda()
    train_y = torch.from_numpy(train_y).to(torch.float32).cuda()
    # 开始加载数据
    torch_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
    # 训练时需要分配训练，测试与验证时不需要
    val_x = torch.from_numpy(val_x).to(torch.float32).cuda()
    val_y = torch.from_numpy(val_y).to(torch.float32).cuda()

    # 只抽一些作为测试
    #data_subject_test, _, label_subject_test, _ = train_test_split(data_subject_test, label_subject_test, test_size=0.2, random_state=2)
    test_x = torch.from_numpy(data_subject_test).to(torch.float32).cuda()
    test_y = torch.from_numpy(label_subject_test).to(torch.float32).cuda()


    """模型训练"""
    model = model_others_20220422.GengNet_id()
    print("Model initialized")
    model.cuda()  # 用GPU进行训练
    torch.save(model, './saved_models/experiment2/identification/%d' % gesture + '_model_2dcnn_emg_base' + '.pt')
    min_valid_loss = 120
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimize all cnn parameters 学习率调小扫平一切牛鬼蛇神
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    curr_patience = patience

    # Learning rate update schedulers  学习率衰减的设置
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #    optimizer, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step
    #)
    #with torch.no_grad():  # 防止训练过程中显存爆炸   # 自己如果加了防止显存爆炸的代码就会不训练
    """训练基线模型"""
    for e in range(epochs):
        model.train()  # pytorch 训练写法
        avg_train_loss = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            """将数据按轴进行分离，组合"""

            b_x = b_x.cuda()
            b_y = b_y.cuda()

            optimizer.zero_grad()  # clear gradients for this training step

            output = model(b_x)  # cnn output
            #x = b_y.long()
            #output.squeeze()
            # 这里是最重要的
            loss = loss_func(output, b_y.long())  # 算源域的损失，这个是有标签的

            loss.requires_grad_(True)  # 生成loss的梯度

            loss.backward()
            optimizer.step()  # apply gradients
            avg_loss = loss.item()
            avg_train_loss += avg_loss
        avg_train_loss = avg_train_loss
        print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        """模型评估（在每一个epoch结束时进行）"""
        model.eval()  # pytorch 评估写法

        # 输入验证数据
        val_output = model(val_x)
        # 计算损失函数
        valid_loss = loss_func(val_output, val_y.long())
        avg_valid_loss = valid_loss.item()
        # 计算输出
        pred_y = torch.max(val_output.cpu(), 1)[1]

        accuracy = accuracy_score(val_y.cpu(), pred_y)
        print('Validation accuracy: %.2f' % accuracy)
        avg_valid_loss = avg_valid_loss #/ len(val_y)
        print("Validation loss is: {}".format(avg_valid_loss))

        if (avg_valid_loss < min_valid_loss):  # 如果当前损失小于之前的最小损失则将模型保存
            curr_patience = patience  # 如果二十步没有提升就结束训练
            min_valid_loss = avg_valid_loss
            torch.save(model, './saved_models/experiment2/identification/%d' % gesture + '_model_2dcnn_emg_base' + '.pt')
            print("Found new best model, save to disk")
        else:
            curr_patience -= 1
        if curr_patience <= 0:
            break
        #lr_scheduler.step()  # 学习率衰减
        # """训练完成后用真实数据进行测试"""
    """
    with torch.no_grad():
        
        best_model = torch.load('./saved_models/experiment2/identification/%d' % gesture + '_model_2dcnn_emg_base' + '.pt')
        best_model.eval()
        # 输入验证数据
        test_output = best_model(test_x)
        # 计算输出
        pred_y = torch.max(test_output.cpu(), 1)[1]
        accuracy = accuracy_score(test_y.cpu(), pred_y)
        recall = recall_score(test_y.cpu(), pred_y, average='macro')
        f1 = f1_score(test_y.cpu(), pred_y, average='macro')
        C = confusion_matrix(test_y.cpu(), pred_y)

        plot_confusion_matrix_bigger(C, labels_name, "confusion matrix " + str(gesture) + " accuracy " + str('%.4f' %accuracy))
        plt.savefig('./saved_models/experiment2/identification/model_2dcnn_emg_base_' + str(gesture) + '.png', format='png')
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

print("ACC of Tf %s" % acc_list)  # 用于储存所有个体的迁移学习的准确率,召唤率与F1
print("Recall of Tf %s" % recall_list)
print("F1 of Tf %s" % f1_list)
print("Accuracy is %s" % (np.sum(np.array(acc_list)) / len(gesture_list)))
print("Recall is %s" % (np.sum(np.array(recall_list)) / len(gesture_list)))
print("F1 is %s" % (np.sum(np.array(f1_list)) / len(gesture_list)))
plot_confusion_matrix_bigger(C_list, labels_name, "Confusion Matrix (Accuracy: " + str('%.4f'% (np.sum(np.array(acc_list)) / len(gesture_list)))+"%)")
plt.savefig('./saved_models/experiment2/identification/model_2dcnn_emg_base.png', format='png')
plt.show()
plt.close()
C_list = np.zeros((num_classes, num_classes))
acc_list = []
recall_list = []
f1_list = []
"""