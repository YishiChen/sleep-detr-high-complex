import torch
import torch.nn 
import torch.nn.Functional 
import torch.optim as optim

## Class defining layers. Input: (600 * sampling frequency )
class my_back(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 6, 5)
        self.pool = nn.MaxPool1d(3, stride=2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_channels = 8
batch_size = 16
input = torch.rand((1, 600*128, num_channels))
model = my_back(num_channels=num_channels)

output = model(input)

print(input.shape, output.shape)