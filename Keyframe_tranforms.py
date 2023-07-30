# # Data augmentation and normalization for training
# # Just normalization for validation

from torchvision import datasets, models, transforms

input_size=224   # for Resnet101

# input_size = 299  for Inception

def keyframe_transform(image):

    train_data_transforms = transforms.Compose ([transforms.ToPILImage () ,
                                      transforms.RandomResizedCrop (input_size) ,
                                      transforms.RandomHorizontalFlip () ,
                                      transforms.ToTensor () ,
                                      transforms.Normalize ([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
                                      ])

    test_data_transforms= transforms.Compose ([transforms.ToPILImage () ,
                                     transforms.Resize (input_size) ,
                                     transforms.CenterCrop (input_size) ,
                                     transforms.ToTensor () ,
                                     transforms.Normalize ([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
                                     ]) ,


