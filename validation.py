import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader,  lstm_model,spatail_model,temporal_model, classifer,criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    temporal_model.eval()
    spatail_model.eval()
    lstm_model.eval()
    classifer.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (clips,key_frame, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets= targets.cuda()
        clips = clips.cuda()
        key_frame = key_frame.cuda()

        # spatail_feature = spatail_model(key_frame)  # (batch_size,512,8,8)
        # spatail_feature = spatail_feature.permute(0, 2, 3, 1)  # (batch_size,8,8,512)
        # spatail_feature = spatail_feature.view(spatail_feature.size(0), -1,
        #                                        spatail_feature.size(-1))  # (batch_size,64,2048)
        # # print(spatail_feature.size())   #output....[64,49,2048]
        # spatail_feature = lstm_model(spatail_feature)
        # # print(spatail_feature.size())  #output.... [64,101]
        #
        # temp_feature = temporal_model(clips)
        # # temp_feature=temp_feature.view(temp_feature.size (0) , temp_feature.size (-1),-1)    #output....[64,1,2048]
        # # print(temp_feature.size())   #output.... [64,101]
        # # print(temp_feature.size())
        #
        # outputs = spatail_feature + temp_feature
        # # outputs = torch.cat((spatail_feature, temp_feature), dim=0)
        # print(outputs.size())
        outputs = classifer(key_frame, clips)
        # outputs = classifer(outputs)
        #
        # print(outputs.size())

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        # print(loss, acc)
        losses.update(loss.data, clips.size(0))
        accuracies.update(acc, clips.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
    print('Validation at epoch {}'.format(epoch), 
        'Loss  ({loss.avg:.4f})\t'
        'Acc  ({acc.avg:.3f})'.format( loss=losses, acc=accuracies))


    return losses.avg, accuracies.avg
