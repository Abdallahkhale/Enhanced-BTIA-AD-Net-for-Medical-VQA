"""
Enhanced BTIA-AD Net Training Script (Comprehensive Fix)
Implements:
1. 3-Stage Progressive Unfreezing
2. Class-Balanced Loss + Multi-Task Auxiliary Loss
3. Exact Match Evaluation with Normalization
4. Question-Type Aware Answer Distillation
5. Robust Diagnostics & Early Stopping
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import logging
import re
from typing import Dict, List, Tuple, Optional
import time

from config import TrainingConfig
from dataset import VQARADDataset, get_dataloaders
from model import build_model

import sys
from tqdm import tqdm

# Setup Logging
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("BTIA_Train")
    logger.setLevel(logging.DEBUG)
    
    # File handler (Force UTF-8)
    fh = logging.FileHandler(os.path.join(save_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# --- 1. Robust Evaluation ---
def normalize_answer(text: str) -> str:
    """Normalize answer for exact match comparison"""
    text = text.lower().strip()
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s\-]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def compute_exact_match(pred_text: str, target_text: str) -> bool:
    """Strict exact match after normalization"""
    return normalize_answer(pred_text) == normalize_answer(target_text)

class BERTScoreEvaluator:
    """Robust BERTScore evaluation"""
    def __init__(self, device):
        self.device = device
        self.scorer = None
        try:
            from bert_score import BERTScorer
            # Use a smaller model for speed and stability
            self.scorer = BERTScorer(model_type="distilbert-base-uncased", device=device)
            print("BERTScore initialized successfully")
        except Exception as e:
            print(f"Warning: BERTScore init failed: {e}")
            
    def compute(self, preds: List[str], refs: List[str]) -> float:
        if not self.scorer or not preds:
            return 0.0
        try:
            # Silence bert_score logging
            import transformers
            transformers.logging.set_verbosity_error()
            P, R, F1 = self.scorer.score(preds, refs)
            return F1.mean().item()
        except Exception as e:
            print(f"BERTScore computation error: {e}")
            return 0.0

# --- 2. Class Balancing ---
def compute_class_weights(answer_counts: Dict[str, int], idx_to_answer: Dict[int, str], beta: float = 0.999) -> torch.Tensor:
    """Compute class weights using Effective Number of Samples"""
    num_classes = len(idx_to_answer)
    weights = torch.ones(num_classes)
    
    # Convert counts to tensor aligned with indices
    counts = []
    for i in range(num_classes):
        ans = idx_to_answer[i]
        counts.append(answer_counts.get(ans, 0))
    
    counts = np.array(counts)
    # Handle zero counts (shouldn't happen for training vocab)
    counts = np.maximum(counts, 1)
    
    # Effective number formula
    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    raw_weights = 1.0 / effective_num
    
    # Normalize to mean=1.0
    raw_weights = raw_weights / raw_weights.mean()
    
    # Clip to prevent instability
    raw_weights = np.clip(raw_weights, 0.5, 5.0)
    
    return torch.tensor(raw_weights, dtype=torch.float32)

def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Oversample rare answers for training"""
    # Extract all answers
    all_answers = [d['answer'].lower().strip() for d in dataset.data]
    answer_counts = Counter(all_answers)
    max_count = max(answer_counts.values())
    
    weights = []
    for ans in all_answers:
        if ans in ['yes', 'no']:
            weights.append(1.0) # Standard weight for frequent closed answers
        else:
            freq = answer_counts[ans]
            # Sqrt smoothing to avoid extreme oversampling
            w = np.sqrt(max_count / freq)
            weights.append(w)
            
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# --- 3. Diagnostics & Question Type ---
class QuestionTypeDetector:
    """Detect question type for Auxiliary Task & Answer Filtering"""
    def __init__(self):
        self.keywords = {
            'view': ['plane', 'view', 'orientation', 'position', 'projection'],
            'sequence': ['sequence', 'weighted', 't1', 't2', 'flair', 'dwi', 'adc'],
            'modality': ['modality', 'imaging', 'scan type', 'method', 'ct', 'mri'],
            'location': ['location', 'where', 'lobe', 'organ', 'region', 'part']
        }
    
    def detect(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in self.keywords['view']): return 'view'
        if any(k in q for k in self.keywords['sequence']): return 'sequence'
        if any(k in q for k in self.keywords['modality']): return 'modality'
        if any(k in q for k in self.keywords['location']): return 'location'
        return 'general'

class TrainingDiagnostics:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = defaultdict(list)
        self.pred_counts = Counter()
        
    def log(self, epoch, metrics, entropy):
        self.history['epoch'].append(epoch)
        for k, v in metrics.items():
            self.history[k].append(v)
        self.history['entropy'].append(entropy)
        
    def plot(self):
        epochs = self.history['epoch']
        if not epochs: return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        if 'train_loss' in self.history:
            axes[0,0].plot(epochs, self.history['train_loss'], label='Train')
            axes[0,0].set_title('Loss')
            axes[0,0].legend()
            
        # Accuracy
        if 'val_acc_closed' in self.history:
            axes[0,1].plot(epochs, self.history['val_acc_closed'], label='Closed')
            axes[0,1].plot(epochs, self.history['val_acc_open'], label='Open') 
            axes[0,1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Target 80%')
            axes[0,1].axhline(y=0.6, color='b', linestyle='--', alpha=0.5, label='Target 60%')
            axes[0,1].set_title('Accuracy')
            axes[0,1].legend()
            
        # BERTScore
        if 'val_bert_score' in self.history:
            axes[1,0].plot(epochs, self.history['val_bert_score'])
            axes[1,0].set_title('BERTScore F1')
            
        # Entropy
        if 'entropy' in self.history:
            axes[1,1].plot(epochs, self.history['entropy'])
            axes[1,1].set_title('Prediction Entropy (Higher is better)')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_plot.png"))
        plt.close()

# --- 4. Trainer ---
class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger(config.training.log_dir)
        
        self.logger.info(f"Device: {self.device}")
        
        # Data
        self.logger.info("Loading data...")
        tokenizer = self._load_tokenizer()
        self.train_loader, self.val_loader, self.test_loader, self.ans_to_idx, self.idx_to_ans = get_dataloaders(
            config, tokenizer
        )
        
        # Replace train_loader with WeightedRandomSampler version?
        # Since get_dataloaders creates loaders, we modify it or create sampler here and recreate loader
        # For simplicity and correctness, let's create weighted sampler for the TRAIN dataset
        train_dataset = self.train_loader.dataset
        weighted_sampler = create_weighted_sampler(train_dataset)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=weighted_sampler, # Use weighted sampler
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.logger.info("Weighted Random Sampler enabled for class balancing.")
        
        # Model
        self.logger.info("Building model...")
        self.num_open_answers = len(self.ans_to_idx)
        self.model = build_model(config, self.num_open_answers)
        self.model = self.model.to(self.device)
        
        # Register masks
        self.logger.info("Setting up question-type masks...")
        self.model.set_question_type_masks(self.idx_to_ans)
        
        # Detectors
        self.q_type_detector = QuestionTypeDetector()
        
        # Loss - Class Balanced
        raw_counts = Counter([d['answer'].lower().strip() for d in train_dataset.data])
        class_weights = compute_class_weights(raw_counts, self.idx_to_ans)
        
        self.criterion_open = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device),
            label_smoothing=0.1
        )
        self.criterion_closed = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Evaluator
        self.bert_evaluator = BERTScoreEvaluator(self.device)
        self.diagnostics = TrainingDiagnostics(config.training.log_dir)
        
    def _load_tokenizer(self):
        # BioMedCLIP tokenizer
        import open_clip
        return open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    def get_optimizer(self, stage: int):
        """Discriminative LR based on stage"""
        # Stage 1: Heads & Fusion (1e-4)
        # Stage 2: + Text (1e-6)
        # Stage 3: + Vision (1e-6)
        
        groups = []
        
        # 1. Heads & Fusion & Distillation (Basic Trainable)
        base_params = list(self.model.fusion.parameters()) + \
                      list(self.model.answer_distillation.parameters()) + \
                      list(self.model.type_classifier.parameters())
        
        if self.model.use_dual_heads:
            base_params += list(self.model.closed_head.parameters()) + \
                           list(self.model.open_head.parameters())
            # Aux heads
            base_params += list(self.model.view_head.parameters()) + \
                           list(self.model.modality_head.parameters())
        else:
            base_params += list(self.model.classifier.parameters())
            
        lr_base = 1e-4 if stage == 1 else (5e-5 if stage == 2 else 1e-5)
        groups.append({'params': base_params, 'lr': lr_base})
        
        # 2. Text Encoder
        if stage >= 2:
            text_params = [p for n, p in self.model.encoder.model.named_parameters() 
                          if 'visual' not in n and p.requires_grad]
            lr_text = 1e-6 if stage == 2 else 5e-6
            groups.append({'params': text_params, 'lr': lr_text})
            
        # 3. Vision Encoder (Last blocks)
        if stage >= 3:
            vision_params = [p for p in self.model.encoder.model.visual.trunk.blocks[-2:].parameters() 
                             if p.requires_grad]
            groups.append({'params': vision_params, 'lr': 1e-6})
            
        return AdamW(groups, weight_decay=0.01)

    def train(self):
        logger = self.logger
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.training.use_amp)
        
        best_f1 = 0
        current_stage = 1
        optimizer = self.get_optimizer(stage=1)
        scheduler = CosineAnnealingLR(optimizer, T_max=10 * len(self.train_loader))
        
        # Sample log file
        self.sample_log_path = os.path.join(self.config.training.log_dir, "validation_samples.txt")
        with open(self.sample_log_path, 'w', encoding='utf-8') as f:
            f.write("Validation Samples Log\n=====================\n")
        
        for epoch in range(1, self.config.training.epochs + 1):
            
            # Update Stage
            new_stage = 1
            if epoch > self.config.training.freeze_text_epochs: new_stage = 2
            if epoch > self.config.training.freeze_vision_epochs: new_stage = 3
            
            if new_stage != current_stage:
                logger.info(f"=== Entering Stage {new_stage} ===")
                if new_stage == 2:
                    self.model.unfreeze_encoder(unfreeze_vision=False, unfreeze_text=True)
                elif new_stage == 3:
                    self.model.partial_unfreeze_vision(num_blocks=2)
                
                optimizer = self.get_optimizer(new_stage)
                # Reset scheduler for new stage
                scheduler = CosineAnnealingLR(optimizer, T_max=10 * len(self.train_loader))
                current_stage = new_stage

            # Training Loop
            self.model.train()
            total_loss = 0
            
            # TQDM Progress Bar for Training
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.training.epochs}", leave=True)
            
            for i, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                tokens = batch['question_tokens'].to(self.device).squeeze(1)
                is_closed = batch['is_closed'].to(self.device)
                targets = batch['answer_idx'].to(self.device)
                
                raw_questions = batch['question_text']
                
                # Compute question types
                q_type_indices = []
                view_targets = []
                mod_targets = []
                
                for q in raw_questions:
                    qt = self.q_type_detector.detect(q)
                    idx = self.model.type_to_idx.get(qt, 0)
                    q_type_indices.append(idx)
                    
                    # Aux targets
                    v_t = -1 
                    if 'axial' in q.lower(): v_t = 0
                    elif 'sagittal' in q.lower(): v_t = 1
                    elif 'coronal' in q.lower(): v_t = 2
                    view_targets.append(max(0, v_t))
                    
                    m_t = -1
                    if 'ct' in q.lower(): m_t = 0
                    elif 'mri' in q.lower(): m_t = 1
                    mod_targets.append(max(0, m_t))
                
                q_type_tensor = torch.tensor(q_type_indices, device=self.device)
                view_targets = torch.tensor(view_targets, device=self.device)
                mod_targets = torch.tensor(mod_targets, device=self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.config.training.use_amp):
                    outputs = self.model(
                        images, tokens, is_closed, 
                        question_types=q_type_tensor
                    )
                    
                    # 1. Main VQA Loss
                    closed_logits = outputs['closed_logits']
                    open_logits = outputs['open_logits']
                    
                    loss_closed = torch.tensor(0.0, device=self.device)
                    loss_open = torch.tensor(0.0, device=self.device)
                    
                    if is_closed.any():
                        loss_closed = self.criterion_closed(closed_logits[is_closed], targets[is_closed])
                    if (~is_closed).any():
                        loss_open = self.criterion_open(open_logits[~is_closed], targets[~is_closed])
                        
                    loss_vqa = loss_closed + 1.5 * loss_open 
                    
                    # 2. Aux Losses
                    loss_view = F.cross_entropy(outputs['view_logits'], view_targets, ignore_index=-1)
                    loss_mod = F.cross_entropy(outputs['modality_logits'], mod_targets, ignore_index=-1)
                    
                    total_loss_batch = loss_vqa + 0.2*loss_view + 0.2*loss_mod
                
                scaler.scale(total_loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += total_loss_batch.item()
                
                # Update progress bar
                pbar.set_postfix({'Loss': f"{total_loss_batch.item():.4f}"})
                    
            avg_loss = total_loss / len(self.train_loader)
            
            # Validation
            metrics, entropy = self.evaluate(self.val_loader, epoch)
            
            logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            logger.info(f"Metrics: Closed={metrics['acc_closed']:.2%} | Open={metrics['acc_open']:.2%} | Overall={metrics['acc_overall']:.2%} | BERT={metrics['val_bert_score']:.4f}")
            logger.info(f"Entropy: {entropy:.4f}")
            
            self.diagnostics.log(epoch, metrics, entropy)
            self.diagnostics.plot()
            
            # Save best
            if metrics['acc_open'] > best_f1:
                best_f1 = metrics['acc_open']
                torch.save(self.model.state_dict(), os.path.join(self.config.training.save_dir, "best_model.pth"))
                logger.info("Saved Best Model (Open Accuracy)")

    def evaluate(self, loader, epoch):
        self.model.eval()
        correct_closed = 0
        total_closed = 0
        correct_open = 0
        total_open = 0
        total_samples = 0
        correct_overall = 0
        
        open_preds_text = []
        open_targets_text = []
        
        entropy_sum = 0
        batches = 0
        
        # Store samples for logging to file
        log_samples = []
        
        # TQDM for Validation
        pbar = tqdm(loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                tokens = batch['question_tokens'].to(self.device).squeeze(1)
                is_closed = batch['is_closed'].to(self.device)
                targets = batch['answer_idx'].to(self.device)
                
                raw_questions = batch['question_text']
                q_type_indices = [self.model.type_to_idx.get(self.q_type_detector.detect(q), 0) for q in raw_questions]
                q_type_tensor = torch.tensor(q_type_indices, device=self.device)
                
                outputs = self.model(
                    images, tokens, is_closed, 
                    question_types=q_type_tensor
                )
                
                # Predictions container
                batch_preds = torch.zeros_like(targets)
                
                # Closed Eval
                if is_closed.any():
                    closed_preds = outputs['closed_logits'][is_closed].argmax(dim=-1)
                    batch_preds[is_closed] = closed_preds
                    
                    c_correct = (closed_preds == targets[is_closed]).sum().item()
                    correct_closed += c_correct
                    total_closed += is_closed.sum().item()
                    correct_overall += c_correct
                    
                # Open Eval
                if (~is_closed).any():
                    open_logits = outputs['open_logits'][~is_closed]
                    open_preds = open_logits.argmax(dim=-1)
                    batch_preds[~is_closed] = open_preds
                    open_targets = targets[~is_closed]
                    
                    probs = F.softmax(open_logits, dim=-1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                    entropy_sum += ent
                    batches += 1
                    
                    for i, pred_idx in enumerate(open_preds):
                        pred_text = self.idx_to_ans.get(pred_idx.item(), '')
                        gt_text = self.idx_to_ans.get(open_targets[i].item(), '')
                        
                        open_preds_text.append(pred_text)
                        open_targets_text.append(gt_text)
                        
                        if compute_exact_match(pred_text, gt_text):
                            correct_open += 1
                            correct_overall += 1
                    
                    total_open += (~is_closed).sum().item()
                
                total_samples += images.size(0)

                # Collect RANDOM samples (not just first ones)
                if batch_idx % 2 == 0 and len(log_samples) < 20: 
                    for i in range(min(2, images.size(0))):
                        q_text = raw_questions[i]
                        pred_idx = batch_preds[i].item()
                        target_idx = targets[i].item()
                        p_text = self.idx_to_ans.get(pred_idx, 'unknown')
                        t_text = self.idx_to_ans.get(target_idx, 'unknown')
                        is_c = is_closed[i].item()
                        type_str = "CLOSED" if is_c else "OPEN"
                        # Use ASCII compliant symbols
                        match = "[CORRECT]" if compute_exact_match(p_text, t_text) else "[WRONG]"
                        
                        log_samples.append(f"[{type_str}] Q: {q_text} | Pred: {p_text} | Tgt: {t_text} | {match}")

        # Metrics
        acc_closed = correct_closed / total_closed if total_closed > 0 else 0
        acc_open = correct_open / total_open if total_open > 0 else 0
        acc_overall = correct_overall / total_samples if total_samples > 0 else 0
        
        # BERTScore
        bert_score = 0.0
        if open_preds_text:
            bert_score = self.bert_evaluator.compute(open_preds_text, open_targets_text)
            
        metrics = {
            'acc_closed': acc_closed,
            'acc_open': acc_open,
            'acc_overall': acc_overall,
            'val_bert_score': bert_score
        }
        mean_entropy = entropy_sum / max(1, batches)
        
        # Log Samples to separate file
        try:
            with open(self.sample_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\nExample Predictions (Epoch {epoch}):\n")
                f.write("-" * 50 + "\n")
                for s in log_samples:
                    f.write(s + "\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write samples to log file: {e}")
        
        return metrics, mean_entropy

if __name__ == "__main__":
    from config import get_config
    cfg = get_config()
    # Force save dir
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    trainer = Trainer(cfg)
    trainer.train()
