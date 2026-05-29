"""
=============================================================
  hybrid_model.py — Hybrid ConvNeXt + Swin Transformer
  ─────────────────────────────────────────────────────
  Branch 1 : ConvNeXt-Tiny   (Modern CNN, SOTA feature extraction)
  Branch 2 : Swin-Tiny       (Hierarchical Transformer)
  Fusion   : Concatenation → FC → 4-class softmax
=============================================================
"""

import torch
import torch.nn as nn
import timm

class AlzheimerHybridModel(nn.Module):
    """
    State-of-the-Art Hybrid Architecture (Image Only).
    """

    def __init__(self, num_classes=4):
        super(AlzheimerHybridModel, self).__init__()
        
        # 1. CNN Branch (ConvNeXt-Tiny)
        # -----------------------------------------------------
        # num_classes=0 removes the classifier and returns global pooled features
        self.cnn = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.cnn_dim = self.cnn.num_features # 768
        
        # 2. Transformer Branch (Swin-Tiny)
        # -----------------------------------------------------
        # num_classes=0 removes the classifier and returns global pooled features
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.swin_dim = self.swin.num_features # 768
        
        # 3. Fusion Head
        # -----------------------------------------------------
        # 768 + 768 = 1536 features
        fusion_dim = self.cnn_dim + self.swin_dim 
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # ConvNeXt Features (B, 768)
        cnn_feat = self.cnn(x)
        
        # Swin Features (B, 768)
        swin_feat = self.swin(x)
        
        # Fusion (B, 1536)
        combined = torch.cat((cnn_feat, swin_feat), dim=1)
        
        # Classification
        logits = self.fusion_head(combined)
        return logits
    
    def freeze_backbones(self):
        """Phase 1: freeze encoders, train only head."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.swin.parameters():
            param.requires_grad = False
        print("  [OK] ConvNeXt + Swin backbones frozen (Phase 1)")

    def unfreeze_backbones(self, unfreeze_fraction=0.3):
        """Phase 2: Smart unfreeze for fine-tuning."""
        
        # 1. Unfreeze Last Stages of ConvNeXt
        # ConvNeXt stages key: "stages.3" is the final block
        for name, param in self.cnn.named_parameters():
            if "stages.3" in name or "head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # 2. Unfreeze Last Stage of Swin Transformer
        for name, param in self.swin.named_parameters():
            if "layers.3" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [OK] Optimally unfrozen (ConvNeXt Stage 3, Swin L3). Trainable: {trainable:,}")

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

if __name__ == "__main__":
    model = AlzheimerHybridModel(num_classes=4)
    print(f"Params: {model.count_params()}")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Output: {y.shape}")
