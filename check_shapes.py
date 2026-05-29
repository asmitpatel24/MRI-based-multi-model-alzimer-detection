import torch
import torch.nn as nn
import timm

def check():
    print("Checking shapes...")
    x = torch.randn(2, 3, 224, 224)
    
    # CNN
    cnn = timm.create_model('resnet18', pretrained=False) # No download needed for shape check
    cnn.fc = nn.Identity()
    out_cnn = cnn(x)
    print(f"CNN (ResNet18) output: {out_cnn.shape}")
    
    # ViT
    vit = timm.create_model('vit_small_patch16_224', pretrained=False)
    vit.head = nn.Identity()
    out_vit = vit(x)
    print(f"ViT (Small) output: {out_vit.shape}")
    
    # Hybrid
    try:
        from hybrid_model import AlzheimerHybridModel
        model = AlzheimerHybridModel(num_classes=4)
        out = model(x)
        print(f"Hybrid Model output: {out.shape}")
    except Exception as e:
        print(f"Hybrid Model failed: {e}")

if __name__ == "__main__":
    check()
