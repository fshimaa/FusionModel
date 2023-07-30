import os
import sys
import json
import numpy as np
import torch
from torch import nn
from confusionresult import drawconfusion
from torch import optim
from torch.optim import lr_scheduler
from models import lstm_fusion, LinearClassifierLayer
from utils import DrawLoss , DrawAccur,readfiles
from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
from torchvision import datasets, models, transforms
device = torch.device ("cuda:0" if torch.cuda.is_available () else "cpu")

def initialize_model( feature_extract=True, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = models.resnet101(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 51)
    input_size = 224
    # print ("initalize the Resnet101 model ")
    return  model_ft , input_size


if __name__ == '__main__':
    valLosses=trainLosses=[]
    valAccu=trainAccu=[]

    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    # print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    # for the spetail_features load the saved model for evaluate
    spatail_model , input_size = initialize_model ()
    spatail_model=torch.load('/home/data/R2D-hmdb.pth')
    modules = list (spatail_model.children())[:-2]
    spatail_model=nn.Sequential (*modules)
    spatail_model=spatail_model.cuda()


    # for the lstm model
    lstm_model = lstm_fusion.LSTM (num_classes=51)
    lstm_model.load_state_dict(torch.load('/home/data/HMDB-R2D.pth'))
    lstm_model=lstm_model.cuda()




    # for the spatial_temporal_feature load the saved model for evaluate
    temporal_model, parameters = generate_model(opt)   ## return the pretrianed model with the finetunning paramters
    temporal_model.load_state_dict(torch.load('/home/data/R3D.pth'))
    temporal_model=temporal_model.cuda()
    # print(temporal_model)


    classifier=LinearClassifierLayer.LinearClassifierLayer(spatail_model,lstm_model,temporal_model)
    classifier=classifier.cuda()
    # print((classifier))


    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        keyframe_data_transforms = transforms.Compose ([transforms.ToPILImage () ,
                                                        transforms.RandomResizedCrop (opt.input_size) ,
                                                        transforms.RandomHorizontalFlip () ,
                                                        transforms.ToTensor () ,
                                                        transforms.Normalize ([0.485 , 0.456 , 0.406] ,
                                                                              [0.229 , 0.224 , 0.225])
                                                        ])
        spatial_transform = Compose ([
            Scale (opt.sample_size) ,
            CenterCrop (opt.sample_size) ,
            ToTensor (opt.norm_value) , norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)   #16 frame
        target_transform = ClassLabel ()


        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform, keyframe_data_transforms)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])


        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        optimizer = optim.SGD(
            classifier.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)


    if not opt.no_val:
        keyframe_data_transforms = transforms.Compose ([transforms.ToPILImage () ,
                                                     transforms.RandomResizedCrop (opt.input_size) ,
                                                     transforms.RandomHorizontalFlip () ,
                                                     transforms.ToTensor () ,
                                                     transforms.Normalize ([0.485 , 0.456 , 0.406] ,
                                                                           [0.229 , 0.224 , 0.225])
                                                     ])
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform =TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform, keyframe_data_transforms)

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        temporal_model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])



    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            loss, accu=train_epoch(i, train_loader,lstm_model ,spatail_model,temporal_model, classifier,criterion, optimizer, opt,
                        train_logger, train_batch_logger)

            trainAccu.append(accu)
            trainLosses.append(loss)
        if not opt.no_val:
            validation_loss, accu = val_epoch(i, val_loader,lstm_model, spatail_model,temporal_model, classifier, criterion, opt,
                                        val_logger)
            
            valAccu.append(accu)
            valLosses.append(validation_loss)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)


    PATH = '/home/data/Class_fusion.pth'   ##"/home/data/temp_model.pth"
    #
    torch.save ( classifier.state_dict ( ) , 'model-parameters.pt' )

    trainLosses, trainAccu = readfiles(os.path.join("/home/data/class_output/", "train.log"))
    valLosses, valAccu = readfiles(os.path.join("/home/data/class_output/", "val.log"))
    DrawLoss(trainLosses, valLosses, opt.n_epochs)
    DrawAccur(trainAccu, valAccu, opt.n_epochs)


    # model = torch.load(PATH)
    # print(model)
    # opening the file in read mode
    my_file = open("/home/data/annotation_dir/HMDB_label.txt", "r")
    # reading the file
    data = my_file.read()
    data_into_list = data.split("\n")
    # printing the data
    print(data_into_list)
    my_file.close()


    drawconfusion(  train_loader , lstm_model , spatail_model , temporal_model , classifier ,
                                criterion ,data_into_list,"train" )


    drawconfusion ( val_loader , lstm_model , spatail_model , temporal_model , classifier ,
                                  criterion ,data_into_list, "val" )


