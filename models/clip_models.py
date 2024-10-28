import torch
import torch.nn as nn
import copy
import clip
from transformers import CLIPModel
from peft import get_peft_model, LoraConfig, TaskType
# class ImageEncoder(torch.nn.Module):
#     def __init__(self, version='ViT-B/32', device='cuda'):
#         super().__init__()
#         self.model, _ = clip.load(version, device=device)
#         # make the model weight

#     def forward(self, images):
#         return self.model.encode_image(images)


# class ClassificationHead(torch.nn.Linear):
#     def __init__(self, 
#                  input_dim=512,
#                  num_classes=10,
#                  normalize=True):
#         self.
#         self.normalize = normalize

#     def forward(self, inputs):
#         if self.normalize:
#             inputs = inputs / inputs.norm(dim=-1, keepdim=True)
#         print(inputs.shape)
#         return super().forward(inputs)


# class LinearCLIP(Module):
#     def __init__(self, model, num_classes=10):
#         super(LinearCLIP, self).__init__()
#         self.model, _ = clip.load('ViT-B/32', device='cpu')
#         self.classifier = nn.Linear(512, num_classes)
#         self.freeze()

#     def featurizer(self, input):
#         input_features = self.model.encode_image(input)
#         input_features = input_features / input_features.norm(dim=-1, keepdim=True)
#         return input_features

#     def forward(self, input, out_feature=False):
#         input_features = self.featurizer(input)
#         logits = self.classifier(input_features)
#         if out_feature:
#             return input_features, logits
#         else:
#             return logits

#     def freeze(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True

class CLIPImageClassifier(nn.Module):
    def __init__(self, 
                 backbone_version,
                 num_classes=10,
                 process_images=True):
        super().__init__()
        if "TinyCLIP" in backbone_version:
            self.image_encoder = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
            self.image_encoder.encode_image = self.image_encoder.get_image_features
        else:
            self.image_encoder = CLIPModel.from_pretrained(backbone_version)
            self.image_encoder = self.image_encoder.float()
            self.image_encoder.encode_image = self.image_encoder.get_image_features
        self.classification_head = torch.nn.Linear(512, num_classes)
        self.process_images = process_images
        # print(self.image_encoder)
        # self.lora_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     inference_mode=False,
        #     r=8,  # rank of LoRA
        #     lora_alpha=16,  # alpha of LoRA
        #     lora_dropout=0.1,  # dropout rate for LoRA
        #     target_modules=[
        #                     "vision_model.encoder.layers.11.self_attn.k_proj",
        #                     "vision_model.encoder.layers.11.self_attn.v_proj",
        #                     "vision_model.encoder.layers.11.self_attn.q_proj",
        #                     # "vision_model.visual_projection"]
        #     ]
        # )
        # self.image_encoder = get_peft_model(self.image_encoder, self.lora_config)
        # print(self.image_encoder)

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder.encode_image(inputs)
        inputs = inputs.float()
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        outputs = self.classification_head(inputs)
        return outputs
    
    def get_last_layer(self):
        return self.classification_head


if __name__ == "__main__":
    model = CLIPImageClassifier().cuda()
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 10, (10,)))
    train_loader = DataLoader(train_dataset, batch_size=2)
    for images, labels in train_loader:
        images = images.cuda()
        outputs = model(images)
        print(outputs.shape)
        break