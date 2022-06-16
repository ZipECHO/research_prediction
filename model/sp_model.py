from torch import nn
from config import *
class sp_model(nn.Module):
    def __init__(self,num_key,key_dim,drop_rate):
        super(sp_model,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_key*key_dim,1024),
            nn.Dropout(drop_rate),
            nn.Linear(1024,512),
            # nn.Dropout(drop_rate),
            nn.Linear(512,256),
            # nn.Dropout(drop_rate),
            nn.Linear(256,160),
        )
    def forward(self,input):
        return self.net(input)