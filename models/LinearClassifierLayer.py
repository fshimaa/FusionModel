from builtins import print

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifierLayer (nn.Module):
    def __init__(self, mode1, mode2, mode3, num_classes=101):
        super(LinearClassifierLayer, self).__init__()
        self.spatial = mode1
        self.lstm = mode2
        self.temp = mode3

        self.linear1 = torch.nn.Linear(51, 101)
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(101, 51)
        # self.softmax = torch.nn.Softmax()

    def forward(self, keyframe, clips):
        spatail_feature = self.spatial(keyframe)  # (batch_size,512,8,8)
        spatail_feature = spatail_feature.permute(0, 2, 3, 1)  # (batch_size,8,8,512)
        spatail_feature = spatail_feature.view(spatail_feature.size(0), -1,
                                               spatail_feature.size(-1))  # (batch_size,64,2048)
        # print(spatail_feature.size())   #output....[64,49,2048]

        out1 = self.lstm(spatail_feature)
        # print(out1.size())  #output.... [64,101]
        # print(out1)
        out2 = self.temp(clips)

        # print("\n \n ",out2.size())  # output.... [64,101]
        # print(out2)
        # x = torch.cat((out1,out2), dim=0)
        x = torch.add ( out1 , out2 )

        # print("\t", x.size(), "\n")

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x


