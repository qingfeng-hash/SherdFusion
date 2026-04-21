import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv3_S_Encoder(nn.Module):
    def __init__(self, weight_path, proj_dim=128, train_backbone=True):
        super().__init__()

        # 1. 创建 ViT-S/16 backbone
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True
        )

        # 2. 加载 DINOv3 权重
        # state = torch.load(weight_path, map_location="cpu")
        # self.backbone.load_state_dict(state, strict=True)

        # 3. 是否冻结 backbone
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 4. 投影头（对比学习标准做法）
        self.projector = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, proj_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)          # [B, 384]
        feat = F.normalize(feat, dim=-1)
        z = self.projector(feat)         # [B, proj_dim]
        z = F.normalize(z, dim=-1)
        return feat, z
