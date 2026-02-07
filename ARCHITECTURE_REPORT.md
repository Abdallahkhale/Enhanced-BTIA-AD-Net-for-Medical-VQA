# Enhanced BTIA-AD Net: Architecture Report

## Executive Summary

Enhanced BTIA-AD Net (Bimodal Transformer for Image Answering with Answer Distillation) is a Medical Visual Question Answering (VQA) system designed for answering clinical questions about medical images. The system leverages **BioMedCLIP** (a domain-specific vision-language model) with custom fusion and classification components.

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph Input
        IMG[Medical Image<br/>224×224×3]
        Q[Question Text]
    end
    
    subgraph BioMedCLIP_Encoder["BioMedCLIP Encoder (Frozen → Fine-tuned)"]
        VE[Vision Encoder<br/>ViT-B/16]
        TE[Text Encoder<br/>PubMedBERT]
    end
    
    subgraph Processing
        SAD[Semantic Answer<br/>Distillation]
        CMF[Cross-Modal<br/>Fusion]
    end
    
    subgraph Classification
        CH[Closed Head<br/>Binary: Yes/No]
        OH[Open Head<br/>Multi-class: 430 answers]
    end
    
    subgraph Output
        ANS[Predicted Answer]
    end
    
    IMG --> VE
    Q --> TE
    VE --> |"Image Features [B,512]"| SAD
    TE --> |"Text Features [B,512]"| SAD
    SAD --> |"Top-K Answers [B,10,768]"| CMF
    VE --> CMF
    TE --> CMF
    CMF --> |"Fused Features [B,768]"| CH
    CMF --> |"Fused Features [B,768]"| OH
    CH --> |"Closed Questions"| ANS
    OH --> |"Open Questions"| ANS
```

---

## Component Details

### 1. BioMedCLIP Encoder

**Purpose**: Extract domain-specific visual and textual representations from medical images and clinical questions.

| Component | Architecture | Output Dimension |
|-----------|--------------|------------------|
| Vision Encoder | ViT-B/16 (PubMed pre-trained) | 512 |
| Text Encoder | PubMedBERT | 512 |
| Image Size | 224 × 224 | - |
| Max Question Length | 77 tokens | - |

```mermaid
flowchart LR
    subgraph Vision_Encoder["Vision Encoder"]
        PE[Patch Embedding<br/>16×16 patches]
        POS[Positional<br/>Encoding]
        TB1[Transformer<br/>Block 1-4]
        TB2[Transformer<br/>Block 5-8]
        TB3[Transformer<br/>Block 9-12]
        CLS[CLS Token<br/>Pooling]
    end
    
    IMG[Image] --> PE --> POS --> TB1 --> TB2 --> TB3 --> CLS --> VF[Visual Features<br/>512D]
```

**Key Features**:
- Pre-trained on PubMed image-text pairs
- Multi-scale feature extraction (blocks 3, 6, 9, 11)
- Progressive unfreezing during training

---

### 2. Semantic Answer Distillation Network

**Purpose**: Select top-K most relevant answer candidates based on visual-question context.

```mermaid
flowchart TB
    VF[Visual Features<br/>B×512] --> ATTN[Visual-Question<br/>Attention]
    TF[Text Features<br/>B×512] --> ATTN
    ATTN --> |"Attended Query"| FUSE[Fusion MLP]
    VF --> FUSE
    FUSE --> |"Fused B×768"| SIM[Similarity<br/>Computation]
    AE[Answer Embeddings<br/>430×768] --> SIM
    SIM --> TOPK[Top-K Selection<br/>K=10]
    TOPK --> TKE[Top-K Embeddings<br/>B×10×768]
```

**Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| K (Top-K) | 10 | Number of answer candidates |
| Hidden Dim | 768 | Internal feature dimension |
| Attention Heads | 8 | Multi-head attention |

---

### 3. Cross-Modal Fusion Module

**Purpose**: Deep bi-directional fusion of visual, textual, and answer candidate features.

```mermaid
flowchart TB
    subgraph Self_Attention["Self-Attention Layer"]
        VSA[Visual<br/>Self-Attention]
        TSA[Text<br/>Self-Attention]
    end
    
    subgraph Cross_Attention["Cross-Attention Layer"]
        V2T[Visual → Text]
        T2V[Text → Visual]
    end
    
    subgraph Answer_Guidance["Answer-Guided Attention"]
        AGA[Answer<br/>Attention]
    end
    
    VF[Visual] --> VSA
    TF[Text] --> TSA
    VSA --> V2T
    TSA --> T2V
    V2T --> ADD[Add + Norm]
    T2V --> ADD
    ADD --> AGA
    TKE[Top-K Answers] --> AGA
    AGA --> FFN[Feed-Forward<br/>Network]
    FFN --> OUT[Fused Output<br/>B×768]
```

**Architecture**:
- 8 attention heads
- 2 fusion layers
- FFN expansion ratio: 4×
- Dropout: 0.1

---

### 4. Dual Classification Heads

**Purpose**: Separate specialized classifiers for closed (yes/no) and open (multi-class) questions.

```mermaid
flowchart TB
    FUSED[Fused Features<br/>B×768] --> ROUTE{Question<br/>Type?}
    
    ROUTE --> |Closed| CH[Closed Head]
    ROUTE --> |Open| OH[Open Head]
    
    subgraph Closed_Head["Closed Head"]
        CL1[Linear: 768→384]
        CGELU[GELU]
        CDO[Dropout 0.1]
        CL2[Linear: 384→2]
    end
    
    subgraph Open_Head["Open Head"]
        OL1[Linear: 768→768]
        OGELU[GELU]
        ODO[Dropout 0.1]
        OL2[Linear: 768→430]
    end
    
    CH --> CL1 --> CGELU --> CDO --> CL2 --> COUT[Yes/No]
    OH --> OL1 --> OGELU --> ODO --> OL2 --> OOUT[Answer Class]
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant I as Image
    participant Q as Question
    participant BE as BioMedCLIP
    participant SAD as Answer Distillation
    participant CMF as Cross-Modal Fusion
    participant CLS as Classifier
    participant A as Answer

    I->>BE: Encode Image
    Q->>BE: Tokenize & Encode
    BE->>SAD: Visual + Text Features
    SAD->>SAD: Compute Answer Similarities
    SAD->>CMF: Top-10 Answer Candidates
    BE->>CMF: Visual + Text Features
    CMF->>CMF: Self + Cross Attention
    CMF->>CMF: Answer-Guided Attention
    CMF->>CLS: Fused Features
    CLS->>A: Closed Head (if yes/no)
    CLS->>A: Open Head (if open-ended)
```

---

## Training Pipeline

```mermaid
flowchart TB
    subgraph Data["Data Pipeline"]
        VQA[VQA-RAD Dataset<br/>3,515 samples]
        TRAIN[Train: 1,797]
        TEST[Test: 451]
    end
    
    subgraph Preprocessing
        AUG[Image Augmentation<br/>Flip, Rotate, Color]
        TOK[Question Tokenization<br/>max_len=77]
        ANS[Answer Encoding<br/>430 classes]
    end
    
    subgraph Training["Training Loop"]
        FWD[Forward Pass]
        LOSS[Loss Computation]
        BACK[Backward Pass]
        OPT[AdamW Optimizer]
        SCHED[Cosine LR Scheduler]
    end
    
    subgraph Eval["Evaluation"]
        ACC[Accuracy<br/>Closed/Open/Overall]
        BERT[BERTScore F1<br/>Semantic Similarity]
    end
    
    VQA --> TRAIN & TEST
    TRAIN --> AUG & TOK & ANS
    AUG --> FWD
    TOK --> FWD
    ANS --> LOSS
    FWD --> LOSS --> BACK --> OPT
    OPT --> SCHED
    TEST --> Eval
```

---

## Loss Function

**Simple Dual-Head Loss with Label Smoothing**:

$$\mathcal{L}_{total} = w_{closed} \cdot \mathcal{L}_{CE}^{closed} + w_{open} \cdot \mathcal{L}_{CE}^{open}$$

| Component | Formula | Weight |
|-----------|---------|--------|
| Closed Loss | CrossEntropy (smoothing=0.05) | 1.0 |
| Open Loss | CrossEntropy (smoothing=0.1) | 1.5 |

---

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Samples per batch |
| Learning Rate | 2e-5 | Initial learning rate |
| Optimizer | AdamW | Weight decay = 0.01 |
| Scheduler | Cosine Annealing | η_min = 1e-7 |
| Epochs | 50 | Maximum training epochs |
| Mixed Precision | FP16 | AMP enabled |

### Progressive Unfreezing Schedule

| Epoch | Action |
|-------|--------|
| 0-2 | All encoders frozen |
| 3+ | Unfreeze text encoder |
| 8+ | Unfreeze last 2 vision blocks |

---

## Model Parameters

| Module | Parameters | Trainable (Initial) |
|--------|------------|---------------------|
| BioMedCLIP Encoder | ~150M | Frozen |
| Answer Distillation | ~5M | Yes |
| Cross-Modal Fusion | ~12M | Yes |
| Classification Heads | ~1M | Yes |
| **Total** | **~168M** | **~18M** |

---

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Closed Accuracy | 80% | Yes/No questions |
| Open Accuracy | 60% | Multi-class questions |
| Overall Accuracy | 70% | Combined |
| BERTScore F1 | 0.75+ | Semantic similarity |

---

## File Structure

```
imp2/
├── config.py          # Configuration dataclasses
├── dataset.py         # VQA-RAD data loading
├── model.py           # Neural network architecture
├── train.py           # Training loop (v5)
├── requirements.txt   # Dependencies
├── data/
│   └── vqa_rad/       # Dataset files
├── checkpoints/       # Model weights
│   ├── best_closed.pth
│   ├── best_open.pth
│   └── best_overall.pth
└── logs/              # Training logs
    └── training_v5_*.log
```

---

## References

1. **BioMedCLIP**: Microsoft's biomedical vision-language model pre-trained on PubMed
2. **VQA-RAD**: Radiology-specific VQA dataset (Lau et al., 2018)
3. **BTIA-AD Net**: Original architecture paper (reference implementation)
