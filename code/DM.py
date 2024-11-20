import torch
from torch import nn
from diffusers import UNet2DModel



class ClassConditionedUnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        num_classes = config["num_classes"]
        class_emb_size = config["class_emb_size"]
        sample_size = config["sample_size"]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        layers_per_block = config["layers_per_block"]
        block_out_channels = config["block_out_channels"]
        down_block_types = config["down_block_types"]
        up_block_types = config["up_block_types"]
        
        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=sample_size,  
            in_channels=in_channels + class_emb_size,  
            out_channels=out_channels,  
            layers_per_block=layers_per_block,  
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dimension
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)

