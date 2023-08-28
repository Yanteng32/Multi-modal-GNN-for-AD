import os
import util
from parameters import TrainType
import torch.nn as nn
from util import *
from matplotlib import pyplot as plt


path_feature = "adnc0530vlg2new.csv"
path_feature_1 = "adnc0530pet2.csv"

path_extra = "adnc0530listplus.csv"

'''
path_feature = "smcipmci2vlg2.csv"
path_feature_1 = "smcipmci2pet2.csv"

path_extra = "smcipmci2listplus.csv"
'''

loss_histories = []
val_acc_histories = []
train_acc_histories = []
vote_train_acc_histories = []
vote_val_acc_histories = []
model_input_arr = []
best_val_accs = []
best_train_accs = []
best_test_accs = []

best_vote_train_acc = 0
best_vote_val_acc = 0
best_vote_test_acc = 0

device = "cuda:1" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss().to(device)

if parameters.train_type == TrainType.Single:
    sexMask, ageMask, sexageMask, apoeMask, mmseMask, mmsapoMask, sexmmsMask, allMask, alloneMask = util.getAgeAndSexMaskFromCSV(path_extra)
    adjacency, features, labels_and_masks = util.prepareData(path_feature, sexMask, device)

    model_input = processData(features, device, adjacency)
    model_input.SetModelType(parameters.ModelType.Cheb)
    model_input_arr.append(model_input)

elif parameters.train_type == TrainType.Mixed:

    sexMask, ageMask, sexageMask, apoeMask, mmseMask, mmsapoMask, sexmmsMask, allMask, alloneMask = util.getAgeAndSexMaskFromCSV(path_extra)
    # 获得邻接矩阵和features, 以及标签
    adjacency_0, features_0, labels_and_masks = util.prepareData(path_feature, mmseMask, device)
    adjacency_1, features_1, labels_and_masks = util.prepareData(path_feature_1, mmseMask, device)

    if parameters.use_dot_product_adjacency:
        adjacency_dot_product = adjacency_0 * adjacency_1
        adjacency_dot_product = adjacency_dot_product * mmseMask
        adjacency_0 = adjacency_dot_product
        adjacency_1 = adjacency_dot_product
    # processData函数处理feature和邻接矩阵
    
    model_input = processData(features_0, device, adjacency_0)
    # 设置想要的模型类别
    model_input.SetModelType(parameters.ModelType.Cheb)
    model_input.SetModelWeight(0.5)
    model_input_arr.append(model_input)


    model_input = processData(features_1, device, adjacency_1)
    model_input.SetModelType(parameters.ModelType.Cheb)
    model_input.SetModelWeight(0.5)
    model_input_arr.append(model_input)

modelsToTrain, optimizers = getModels(model_input_arr, device)

for md in modelsToTrain:
    md.train()
    loss_histories.append([])
    val_acc_histories.append([])
    train_acc_histories.append([])
    best_val_accs.append(0)
    best_train_accs.append(0)
    best_test_accs.append(0)

for epoch in range(parameters.epochs):
    test_logits = []
    val_logits = []
    train_logits = []
    for index, model in enumerate(modelsToTrain):
        logits = modelForward(model, model_input_arr[index])
        train_mask_logits = logits[labels_and_masks.train_mask]

        loss = criterion(train_mask_logits, labels_and_masks.getTrainY())  # 计算损失值
        optimizers[index].zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizers[index].step()  # 使用优化方法进行梯度更新

        train_acc, _ = test(model, labels_and_masks.train_mask, labels_and_masks.labels, model_input_arr[index])
        val_acc, val_mask_logits = test(model, labels_and_masks.val_mask, labels_and_masks.labels, model_input_arr[index])
        test_acc, test_mask_logits = test(model, labels_and_masks.test_mask, labels_and_masks.labels, model_input_arr[index])
        train_logits.append((train_mask_logits, model_input_arr[index].model_weight))
        val_logits.append((val_mask_logits, model_input_arr[index].model_weight))
        test_logits.append((test_mask_logits, model_input_arr[index].model_weight))

        # 计算训练过程中损失值和准确率的变化，用于画图
        loss_histories[index].append(loss.item())
        val_acc_histories[index].append(val_acc.item())
        train_acc_histories[index].append(train_acc.item())
        vac = val_acc.item()
        trac = train_acc.item()
        tac = test_acc.item()
        print(
            "Model {:} Epoch {:03d}, TrainAcc {:.4}, ValAcc {:.4f}, TestAcc {:4f}, Loss {:.4f}".format(type(model),
                                                                                                       epoch,
                                                                                                       trac,
                                                                                                       vac,
                                                                                                       tac,
                                                                                                       loss.item()))
        # 单模型推断, 满足条件提前退出
        if len(modelsToTrain) == 1:
            if earlyStop(epoch, best_train_accs[0], best_val_accs[0], trac, vac):
                break

        if epoch > parameters.minimum_epochs:
            if vac >= best_val_accs[index]:
                best_val_accs[index] = vac
            if trac >= best_train_accs[index]:
                best_train_accs[index] = trac
            if tac >= best_test_accs[index]:
                best_test_accs[index] = tac

    # 训练了两个模型, 进行投票
    if len(modelsToTrain) == 2:
        vote_tain_acc, _ = votingConference(train_logits, labels_and_masks.getTrainY())
        vote_val_acc, _ = votingConference(val_logits, labels_and_masks.getValY())
        vote_test_acc, test_pred = votingConference(test_logits, labels_and_masks.getTestY())
        print("voting probabilities:Epoch: {}, training: {:4f}, validation: {:4f}, testing {:4f}".format(epoch,
                                                                                                         vote_tain_acc,
                                                                                                         vote_val_acc,
                                                                                                         vote_test_acc))

        vote_train_acc_histories.append(vote_tain_acc.item())
        vote_val_acc_histories.append(vote_val_acc.item())
        label_y = labels_and_masks.getTestY()
        true_label = []
        data_pre = []
        true_label.extend(list(label_y.cpu().flatten().numpy()))
        data_pre.extend(list(test_pred.cpu().flatten().numpy()))
        confusionMatrix(true_label, data_pre)

        if earlyStop(epoch, best_vote_train_acc, best_vote_val_acc, vote_tain_acc, vote_val_acc):
            break

        if epoch > parameters.minimum_epochs:
            if vote_tain_acc >= best_vote_train_acc:
                best_vote_train_acc = vote_tain_acc
            if vote_val_acc >= best_vote_val_acc:
                best_vote_val_acc = vote_val_acc
            if vote_test_acc >= best_vote_test_acc:
                best_vote_test_acc = vote_test_acc

plotFolder = './adj_dot_product_plots'
os.makedirs(plotFolder, exist_ok=True)
colors = ['pink', 'purple']

plt.xlabel('epoch')
plt.ylabel('loss')
title = 'trianing losses'
plt.title(title)
for ind in range(len(loss_histories)):
    loss_history = loss_histories[ind]
    epochi = range(0, len(loss_history))
    plt.plot(epochi, loss_history, color=colors[ind], label="model" + str(ind),
             linewidth=1.2)  # 绘制，指定颜色、标签、线宽，标签采用latex格式

plt.legend(loc="upper left")
plt.savefig(plotFolder + '/' + title + '.png')
# Show the plot in non-blocking mode
plt.show(block=False)

plt.xlabel('epoch')
plt.ylabel('accuracy')
title = 'validation accuracies'
plt.title(title)
for ind in range(len(val_acc_histories)):
    val_acc = val_acc_histories[ind]
    epochi = range(0, len(val_acc))
    plt.plot(epochi, val_acc, color=colors[ind], label="model" + str(ind), linewidth=1.0)  # 绘制，指定颜色、标签、线宽，标签采用latex格式

epochi = range(0, len(vote_val_acc_histories))
plt.plot(epochi, vote_val_acc_histories, color='green', label="decision", linewidth=1.0)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
plt.legend(loc="upper left")
plt.savefig(plotFolder + '/' + title + '.png')
plt.show(block=False)


plt.xlabel('epoch')
plt.ylabel('accuracy')
title = 'training accuracies'
plt.title(title)

for ind in range(len(train_acc_histories)):
    train_acc = train_acc_histories[ind]
    epochi = range(0, len(train_acc))
    plt.plot(epochi, train_acc, color=colors[ind], label="model" + str(ind), linewidth=1.0)  # 绘制，指定颜色、标签、线宽，标签采用latex格式

#epochi = range(0, len(vote_train_acc_histories))
#plt.plot(epochi, vote_train_acc_histories, color='green', label="voting", linewidth=1.0)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
plt.legend(loc="upper left")
plt.savefig(plotFolder + '/' + title + '.png')
plt.show(block=False)

