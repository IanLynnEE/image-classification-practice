
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torchvision.models as models

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        x = self.conv1(x)
        x = self.relu(x)
        out = x

        return out


        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)

        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        self.conv1 = nn.Sequential(nn.Conv2d(64,128,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(128,256,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.residual_block1 = nn.Sequential(residual_block(256))
        self.residual_block2 = nn.Sequential(residual_block(256))

        self.fc1 = nn.Sequential(nn.Linear(6400,16), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        x = self.stem_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = x
        
        return out

class TrainingModel(nn.Module):
    def __init__(self, num_out=10):
        super(TrainingModel, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True, progress=True)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000,num_out),
        )
        

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x