import torch.nn as nn
from progetto.mamba3 import Mamba3LMHeadModel

class Mamba3MultiClass(nn.Module):
    def __init__(self, args, input_dim, num_classes):
        super().__init__()
        self.mamba = Mamba3LMHeadModel(args)
        self.input_projection = nn.Linear(input_dim, args.d_model)
        
        self.classifier = nn.Linear(args.d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        
        for layer in self.mamba.backbone.layers:
            y, _ = layer.mixer(layer.mixer_norm(x))
            x = y + x
            x = x + layer.mlp(layer.mlp_norm(x))
            
        x = self.mamba.backbone.norm_f(x)
        
        last_state = x[:, -1, :] 
        return self.classifier(last_state)