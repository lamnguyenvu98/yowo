import torch
import torch.nn as nn
import os
from CFAM import CFAMBlock

class YOWO(nn.Module):

    def __init__(self,
                 cfg,
                 num_classes: int,
                 n_anchors,
                 backbone_2d_type: str = "darknet",
                 backbone_3d_type: str = "resnext101",
                 weight_pretrained_2d: str = None,
                 weight_pretrained_3d: str = None):
        super(YOWO, self).__init__()

        ##### 2D Backbone #####
        if cfg.MODEL.BACKBONE_2D == "darknet":
            self.backbone_2d = darknet.Darknet(os.path.abspath("cfg/yolo.cfg"))
            num_ch_2d = 425 # Number of output channels for backbone_2d
        else:
            raise ValueError("Wrong backbone_2d model is requested. Please select\
                              it from [darknet]")
        if cfg.WEIGHTS.BACKBONE_2D:# load pretrained weights on COCO dataset
            self.backbone_2d.load_weights(cfg.WEIGHTS.BACKBONE_2D)

        ##### 3D Backbone #####
        if cfg.MODEL.BACKBONE_3D == "resnext101":
            self.backbone_3d = resnext.resnext101()
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "resnet18":
            self.backbone_3d = resnet.resnet18(shortcut_type='A')
            num_ch_3d = 512 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "resnet50":
            self.backbone_3d = resnet.resnet50(shortcut_type='B')
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "resnet101":
            self.backbone_3d = resnet.resnet101(shortcut_type='B')
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "mobilenet_2x":
            self.backbone_3d = mobilenet.get_model(width_mult=2.0)
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "mobilenetv2_1x":
            self.backbone_3d = mobilenetv2.get_model(width_mult=1.0)
            num_ch_3d = 1280 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "shufflenet_2x":
            self.backbone_3d = shufflenet.get_model(groups=3,   width_mult=2.0)
            num_ch_3d = 1920 # Number of output channels for backbone_3d
        elif cfg.MODEL.BACKBONE_3D == "shufflenetv2_2x":
            self.backbone_3d = shufflenetv2.get_model(width_mult=2.0)
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        else:
            raise ValueError("Wrong backbone_3d model is requested. Please select it from [resnext101, resnet101, \
                             resnet50, resnet18, mobilenet_2x, mobilenetv2_1x, shufflenet_2x, shufflenetv2_2x]")
        if cfg.WEIGHTS.BACKBONE_3D:# load pretrained weights on Kinetics-600 dataset
            self.backbone_3d = self.backbone_3d.to(cfg.DEVICE)
            self.backbone_3d = nn.DataParallel(self.backbone_3d, device_ids=None) # Because the pretrained backbone models are saved in Dataparalled mode
            pretrained_3d_backbone = torch.load(cfg.WEIGHTS.BACKBONE_3D)
            backbone_3d_dict = self.backbone_3d.state_dict()
            pretrained_3d_backbone_dict = {k: v for k, v in pretrained_3d_backbone['state_dict'].items() if k in backbone_3d_dict} # 1. filter out unnecessary keys
            backbone_3d_dict.update(pretrained_3d_backbone_dict) # 2. overwrite entries in the existing state dict
            self.backbone_3d.load_state_dict(backbone_3d_dict) # 3. load the new state dict
            self.backbone_3d = self.backbone_3d.module # remove the dataparallel wrapper

        ##### Attention & Final Conv #####
        self.cfam = CFAMBlock(num_ch_2d+num_ch_3d, 1024)
        self.conv_final = nn.Conv2d(1024, 5*(cfg.MODEL.NUM_CLASSES+4+1), kernel_size=1, bias=False)

        self.seen = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input shape (batch, channel, sequence, H, W)
        x_3d = input # Input clip
        x_2d = input[:, :, -1, :, :] # Last frame of the clip that is read

        x_2d = self.backbone_2d(x_2d)
        x_3d = self.backbone_3d(x_3d) # # (batch, 2048, 1, h, w)

        x_3d = torch.squeeze(x_3d, dim=2)

        x = torch.cat((x_3d, x_2d), dim=1)
        x = self.cfam(x) # (batch, 1024, h, w)
        out = self.conv_final(x)
        out = out.view(out.size(0), self.n_anchors, -1, out.size(2), out.size(3))
        out = out.permute(0, 1, 3, 4, 2)
        return out

# yowo = YOWO(cfg).to(cfg.DEVICE)

# out = yowo(torch.randn(1, 3, 16, 224, 224).to(device))
# out.shape : torch.Size([1, num_anchors, 7, 7, 5 + num_classes])
