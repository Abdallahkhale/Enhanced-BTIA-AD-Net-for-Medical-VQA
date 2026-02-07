# BTIA-AD Net: Complete Research & Implementation Guide
## Medical Visual Question Answering with Answer Distillation

---

# 🔹 STEP 1 — PAPER SUMMARY

## Problem Being Solved
- **Medical VQA Challenge**: Algorithms struggle with open-ended questions due to large answer vocabulary
- **Open vs Closed Gap**: Open-ended accuracy significantly lower than closed-ended (yes/no) questions
- **Answer Space Explosion**: Hundreds of possible answers make classification difficult

## Core Ideas

### Answer Distillation (AD)
- **Purpose**: Compress large answer space into manageable candidates
- **Method**: Uses visual-guided question attention to select Top-K candidate answers
- **Effect**: Converts open-ended questions into multiple-choice format

### Bi-Text-Image Attention (BTIA)
- **Self-Attention**: Applied within image and text features separately
- **Guided Attention**: Cross-modal interaction between image and text
- **Answer-Guided**: Final fusion incorporates candidate answer information

## Why ResNet-152 and BioBERT?
| Component | Choice | Reason |
|-----------|--------|--------|
| Vision | ResNet-152 | Pre-trained on ImageNet, deep feature extraction |
| Text | BioBERT | Pre-trained on biomedical literature (PubMed) |

## Main Results on VQA-RAD
| Metric | Score |
|--------|-------|
| Open-ended Accuracy | ~61.7% |
| Closed-ended Accuracy | ~74.3% |
| Overall Accuracy | ~69.3% |

## Limitations
- ResNet-152 not trained on medical images
- Limited multi-scale feature extraction
- Answer distillation lacks semantic awareness
- Attention mechanism could be more sophisticated

---

# 🔹 STEP 2 — TECHNIQUES EXPLANATION

## 1. Answer Distillation

### Visual-Guided Question Attention
```
Q_attended = Softmax(Q · V^T / √d) · V
```
- Question features attend to visual features
- Creates visually-grounded question representation

### Softmax Over Answers
```
P(a|Q,V) = Softmax(W · [Q_attended; V_pooled])
```
- Probability distribution over all possible answers
- Uses combined question-visual representation

### Top-K Candidate Selection
```
Candidates = TopK(P(a|Q,V), K=10)
```
- Select K=10 most likely answers
- Reduces answer space from hundreds to 10

### Why K=10 Works Best
| K Value | Accuracy | Coverage |
|---------|----------|----------|
| K=5 | Lower | May miss correct answer |
| K=10 | Optimal | Good balance |
| K=20 | Similar | Diminishing returns |

## 2. Bi-Text-Image Attention

### Self-Attention (Image)
```python
V_self = MultiHeadAttention(V, V, V)  # Image attends to itself
```

### Self-Attention (Question)
```python
Q_self = MultiHeadAttention(Q, Q, Q)  # Text attends to itself
```

### Guided Attention
```python
V_guided = MultiHeadAttention(V_self, Q_self, Q_self)  # Image guided by text
Q_guided = MultiHeadAttention(Q_self, V_self, V_self)  # Text guided by image
```

### Answer-Guided Fusion
```python
Final = Fusion(V_guided, Q_guided, Answer_embeddings)
```

## Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Reduces answer space effectively | ResNet not medical-specific |
| Bi-directional attention | No semantic answer clustering |
| End-to-end trainable | Limited multi-scale features |

---

# 🔹 STEP 3 — VQA-RAD DATASET

## Dataset Statistics
| Attribute | Value |
|-----------|-------|
| Total Images | 315 |
| Total QA Pairs | 3,515 |
| Train QA Pairs | 3,064 |
| Test QA Pairs | 451 |

## Image Types
- **CT Scans**: ~33%
- **MRI**: ~33%
- **X-Ray**: ~33%

## Question Types
| Type | Description | Difficulty |
|------|-------------|------------|
| Closed | Yes/No answers | Easier |
| Open | Free-form answers | Harder |

### Question Categories
- Modality, Plane, Organ, Abnormality, Object, Attribute, Color, Size, Position, Count

## Why Open-Ended is Harder
1. Large vocabulary (hundreds of unique answers)
2. Class imbalance (some answers appear once)
3. Requires precise medical knowledge

## Download Instructions

### Option 1: Hugging Face
```bash
pip install datasets
```

```python
from datasets import load_dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")
```

### Option 2: Manual Download
1. Visit: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
2. Download files
3. Extract to project folder

## Suggested Folder Structure
```
project/
├── data/
│   └── vqa_rad/
│       ├── images/
│       │   ├── synpic100132.jpg
│       │   └── ...
│       ├── train.json
│       └── test.json
├── src/
└── models/
```

## Python Code: Data Loading & Analysis

```python
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
def load_vqa_rad(data_dir):
    with open(os.path.join(data_dir, 'train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, 'test.json'), 'r') as f:
        test_data = json.load(f)
    return train_data, test_data

# Visualize samples
def visualize_samples(data, image_dir, n=4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < len(data):
            sample = data[idx]
            img = Image.open(os.path.join(image_dir, sample['image']))
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Q: {sample['question']}\nA: {sample['answer']}", wrap=True)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('vqa_rad_samples.png')
    plt.show()

# Count question types
def analyze_questions(data):
    q_types = Counter([d.get('answer_type', 'unknown') for d in data])
    print("Question Types Distribution:")
    for qtype, count in q_types.most_common():
        print(f"  {qtype}: {count}")
    return q_types

# Analyze answer distribution
def analyze_answers(data):
    answers = Counter([d['answer'].lower() for d in data])
    print(f"\nTotal unique answers: {len(answers)}")
    print("\nTop 20 answers:")
    for ans, count in answers.most_common(20):
        print(f"  '{ans}': {count}")
    return answers

# Usage
if __name__ == "__main__":
    train, test = load_vqa_rad('data/vqa_rad')
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    analyze_questions(train)
    analyze_answers(train)
```

---

# 🔹 STEP 4 — IMPROVED MODEL DESIGN (RTX 5070 Ti)

## 1. Vision Encoder: BioMedCLIP (Recommended)

### Why BioMedCLIP over ResNet-152?
| Aspect | ResNet-152 | BioMedCLIP |
|--------|------------|------------|
| Pre-training | ImageNet (natural images) | Medical images + text |
| Medical Knowledge | None | Extensive |
| Feature Quality | Good | Excellent for medical |
| Memory | ~60M params | ~86M params (ViT-B/16) |

### Multi-Level Feature Extraction
```python
import torch
from open_clip import create_model_from_pretrained

class BioMedCLIPVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.visual = self.model.visual
        
    def forward(self, x):
        # Get intermediate features for multi-scale
        features = self.visual.trunk.patch_embed(x)
        
        multi_scale = []
        for i, block in enumerate(self.visual.trunk.blocks):
            features = block(features)
            if i in [3, 6, 9, 11]:  # Extract at multiple depths
                multi_scale.append(features)
        
        return multi_scale, features
```

## 2. Text Encoder: PubMedBERT (via BioMedCLIP)

### Justification
- Already integrated in BioMedCLIP
- Pre-trained on PubMed abstracts
- Understands medical terminology

```python
class BioMedCLIPText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
    def forward(self, text_tokens):
        return self.model.encode_text(text_tokens)
```

## 3. Improved Answer Distillation

### Semantic-Aware Distillation
```python
class SemanticAnswerDistillation(torch.nn.Module):
    def __init__(self, ans_embeddings, k=10):
        super().__init__()
        self.k = k
        self.ans_embeddings = ans_embeddings  # Pre-computed medical embeddings
        self.projection = torch.nn.Linear(768, 768)
        
    def forward(self, visual_feat, question_feat):
        # Fused representation
        fused = self.projection(visual_feat + question_feat)
        
        # Semantic similarity with answer embeddings
        similarity = torch.matmul(fused, self.ans_embeddings.T)
        
        # Top-K selection
        topk_scores, topk_indices = torch.topk(similarity, self.k, dim=-1)
        topk_embeddings = self.ans_embeddings[topk_indices]
        
        return topk_scores, topk_indices, topk_embeddings
```

## 4. Fusion Module: Cross-Modal Transformer

```python
class CrossModalFusion(torch.nn.Module):
    def __init__(self, dim=768, heads=8, layers=2):
        super().__init__()
        self.cross_attn_layers = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(dim, heads, batch_first=True)
            for _ in range(layers)
        ])
        self.classifier = torch.nn.Linear(dim, 10)  # K candidates
        
    def forward(self, visual, text, answer_candidates):
        # Visual attends to text
        fused = visual
        for layer in self.cross_attn_layers:
            fused = layer(fused, text)
        
        # Answer-guided attention
        answer_attn = torch.softmax(
            torch.matmul(fused, answer_candidates.transpose(-1, -2)), dim=-1
        )
        output = torch.matmul(answer_attn, answer_candidates)
        
        return self.classifier(output.mean(dim=1))
```

## Complete Architecture

```python
class EnhancedBTIANet(torch.nn.Module):
    def __init__(self, num_answers, k=10):
        super().__init__()
        self.vision_encoder = BioMedCLIPVision()
        self.text_encoder = BioMedCLIPText()
        self.answer_distillation = SemanticAnswerDistillation(k=k)
        self.fusion = CrossModalFusion()
        self.final_classifier = torch.nn.Linear(768, num_answers)
        
    def forward(self, image, question_tokens):
        # 1. Encode
        multi_scale, visual_feat = self.vision_encoder(image)
        text_feat = self.text_encoder(question_tokens)
        
        # 2. Answer Distillation
        topk_scores, topk_idx, topk_emb = self.answer_distillation(
            visual_feat.mean(1), text_feat
        )
        
        # 3. Fusion
        fused = self.fusion(visual_feat, text_feat, topk_emb)
        
        # 4. Final prediction
        logits = self.final_classifier(fused)
        return logits, topk_idx
```

---

# 🔹 STEP 5 — TRAINING PIPELINE (First 4 Epochs)

## Training Configuration
```python
config = {
    'batch_size': 16,        # RTX 5070 Ti 16GB
    'learning_rate': 2e-5,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'epochs': 50,
    'warmup_epochs': 2,
}
```

## Epoch-by-Epoch Breakdown

### Epoch 1: Foundation
| Aspect | Details |
|--------|---------|
| Frozen | Vision encoder (except last 2 blocks) |
| Training | Text encoder, Fusion, Classifier |
| Learning Rate | 2e-5 (with warmup) |
| Expected Loss | 2.5 → 1.8 |
| Behavior | Model learns basic correlations |

### Epoch 2: Warming Up
| Aspect | Details |
|--------|---------|
| Frozen | Vision encoder (except last 4 blocks) |
| Training | More vision layers unfrozen |
| Learning Rate | 2e-5 (peak) |
| Expected Loss | 1.8 → 1.2 |
| Behavior | Cross-modal alignment improving |

### Epoch 3: Learning Patterns
| Aspect | Details |
|--------|---------|
| Frozen | Only first 6 vision blocks |
| Training | Most of the network |
| Learning Rate | 2e-5 |
| Expected Val Acc | ~45-55% |
| Behavior | Answer distillation refining |

### Epoch 4: Consolidation
| Aspect | Details |
|--------|---------|
| Frozen | First 4 vision blocks only |
| Training | Nearly full network |
| Learning Rate | 1.5e-5 (start decay) |
| Expected Val Acc | ~55-65% |
| Signs of Good Learning | Loss decreasing, val acc rising |
| Overfitting Signs | Val loss increasing, train acc >> val acc |

## Loss Function
```python
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        
    def forward(self, logits, topk_scores, targets, topk_targets):
        # Classification loss
        cls_loss = self.ce_loss(logits, targets)
        
        # Distillation ranking loss
        dist_loss = self.ce_loss(topk_scores, topk_targets)
        
        return self.alpha * cls_loss + (1 - self.alpha) * dist_loss
```

---

# 🔹 STEP 6 — END-TO-END MODEL PROCESS

## Step-by-Step Flow

```
1. INPUT
   ├── Image: [B, 3, 224, 224]
   └── Question: "What organ is shown?"

2. VISION ENCODER (BioMedCLIP)
   ├── Patch Embedding: [B, 196, 768]
   ├── Transformer Blocks
   └── Output: [B, 196, 768] + multi-scale features

3. TEXT ENCODER (PubMedBERT)
   ├── Tokenization: [B, max_len]
   ├── BERT Processing
   └── Output: [B, 768]

4. ANSWER DISTILLATION
   ├── Fuse visual + text: [B, 768]
   ├── Compute similarity to all answers
   ├── Select Top-K=10 candidates
   └── Output: topk_indices, topk_embeddings [B, 10, 768]

5. MULTIMODAL FUSION
   ├── Cross-attention (visual ↔ text)
   ├── Answer-guided attention
   └── Output: [B, 768]

6. FINAL PREDICTION
   ├── Classifier: [B, 768] → [B, num_answers]
   ├── Softmax
   └── Output: Predicted answer
```

---

# 🔹 STEP 7 — INFERENCE & EVALUATION

## Loading Trained Model
```python
import torch

def load_model(checkpoint_path, num_answers):
    model = EnhancedBTIANet(num_answers=num_answers)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.cuda()
```

## Single Inference
```python
from PIL import Image
import open_clip

def predict(model, image_path, question, tokenizer, preprocess, idx_to_answer):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).cuda()
    
    # Tokenize question
    question_tokens = tokenizer([question]).cuda()
    
    # Inference
    with torch.no_grad():
        logits, topk_idx = model(image_tensor, question_tokens)
        pred_idx = logits.argmax(dim=-1).item()
        
    return idx_to_answer[pred_idx]

# Usage
model = load_model('best_model.pth', num_answers=500)
answer = predict(model, 'chest_xray.jpg', 'Is there pneumonia?', 
                 tokenizer, preprocess, idx_to_answer)
print(f"Predicted: {answer}")
```

## Batch Evaluation
```python
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, test_loader, idx_to_answer):
    model.eval()
    correct_open = 0
    correct_closed = 0
    total_open = 0
    total_closed = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].cuda()
            questions = batch['question_tokens'].cuda()
            answers = batch['answer_idx'].cuda()
            is_closed = batch['is_closed']
            
            logits, _ = model(images, questions)
            preds = logits.argmax(dim=-1)
            
            for i in range(len(preds)):
                if is_closed[i]:
                    total_closed += 1
                    if preds[i] == answers[i]:
                        correct_closed += 1
                else:
                    total_open += 1
                    if preds[i] == answers[i]:
                        correct_open += 1
    
    print(f"Open-ended Accuracy: {100*correct_open/total_open:.2f}%")
    print(f"Closed-ended Accuracy: {100*correct_closed/total_closed:.2f}%")
    print(f"Overall Accuracy: {100*(correct_open+correct_closed)/(total_open+total_closed):.2f}%")
```

## Attention Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, image, question, preprocess, tokenizer):
    image_tensor = preprocess(image).unsqueeze(0).cuda()
    question_tokens = tokenizer([question]).cuda()
    
    # Get attention weights (requires model modification to return attention)
    with torch.no_grad():
        logits, topk_idx, attention_weights = model(
            image_tensor, question_tokens, return_attention=True
        )
    
    # Reshape to image grid (14x14 for ViT-B/16 with 224 input)
    attn = attention_weights[0].mean(0).reshape(14, 14).cpu().numpy()
    attn = np.resize(attn, (224, 224))
    
    # Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[1].imshow(attn, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[2].imshow(image)
    axes[2].imshow(attn, alpha=0.5, cmap='jet')
    axes[2].set_title('Overlay')
    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    plt.show()
```

---

# Summary Comparison

| Aspect | Original BTIA-AD Net | Improved Model |
|--------|---------------------|----------------|
| Vision | ResNet-152 (ImageNet) | BioMedCLIP ViT (Medical) |
| Text | BioBERT | PubMedBERT (via BioMedCLIP) |
| Features | Single-scale | Multi-scale |
| Distillation | Score-based | Semantic embeddings |
| Fusion | BTIA | Cross-Modal Transformer |
| Memory | ~200M params | ~180M params |
| Expected Accuracy | ~69% | ~75%+ (estimated) |

---

*Guide created for RTX 5070 Ti optimization. Batch size 16 recommended for 16GB VRAM.*
