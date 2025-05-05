import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, dropout_prob=0.3, dropout_prob2=0.3): 
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    
        self.dropout1 = nn.Dropout(p=dropout_prob)

        conv_output_size = config.IMAGE_SIZE // 8
        # flattened_size = 256 * conv_output_size * conv_output_size 
        flattened_size = 128 * conv_output_size * conv_output_size 
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)  
        #self.fc3 = nn.Linear(256, num_classes) 


        # Dropouts after FCs
        self.dropout1 = nn.Dropout(p=dropout_prob) 
        self.dropout2 = nn.Dropout(p=dropout_prob2) 
        # self.dropout2 = nn.Dropout(p=0.4) 


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x))) 

        x = self.dropout1(x) 

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        #x = self.dropout1(x) 

        #x = F.relu(self.fc2(x))
        x = self.dropout2(x) 

        # x = self.fc3(x) #
        x = self.fc2(x) 

        return x

if __name__ == '__main__':
    model = SimpleCNN()
    print(model)

    dummy_input = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    output = model(dummy_input)
    print("\nOutput shape:", output.shape) 
