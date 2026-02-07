import os
import torch
import torch.nn.functional as F
import logging
import sys
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

from config import get_config
from dataset import get_dataloaders
from model import build_model
from train import setup_logger, compute_exact_match, BERTScoreEvaluator, QuestionTypeDetector

def test():
    # 1. Config & Setup
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Logger
    logger = setup_logger(os.path.join(cfg.training.log_dir, "test_results"))
    logger.info(f"Testing on device: {device}")
    
    # 2. Data
    logger.info("Loading data...")
    # Tokenizer
    import open_clip
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # We use the val_loader (2nd return) as the test set because provided "test_loader" (3rd) 
    # refers to the original VQA-RAD test set which was merged into training!
    # The 2nd return is the strict 20% holdout from the 80/20 split.
    _, test_loader, _, ans_to_idx, idx_to_ans = get_dataloaders(cfg, tokenizer)
    
    # 3. Model
    logger.info("Building model...")
    num_open_answers = len(ans_to_idx)
    model = build_model(cfg, num_open_answers)
    model = model.to(device)
    
    # Setup Masks & Detectors (Must be done BEFORE loading state_dict if buffer is saved)
    model.set_question_type_masks(idx_to_ans)
    
    # 4. Load Checkpoint
    checkpoint_path = os.path.join(cfg.training.save_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return
        
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    q_type_detector = QuestionTypeDetector()
    bert_evaluator = BERTScoreEvaluator(device)
    
    # 5. Evaluation Loop
    logger.info("Starting Evaluation on Test Set...")
    
    correct_closed = 0
    total_closed = 0
    correct_open = 0
    total_open = 0
    total_samples = 0
    correct_overall = 0
    
    all_preds_text = []
    all_targets_text = []
    results = []
    
    pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            tokens = batch['question_tokens'].to(device).squeeze(1)
            is_closed = batch['is_closed'].to(device)
            targets = batch['answer_idx'].to(device)
            raw_questions = batch['question_text']
            image_names = batch['image_name']
            
            # Question Types
            q_type_indices = [model.type_to_idx.get(q_type_detector.detect(q), 0) for q in raw_questions]
            q_type_tensor = torch.tensor(q_type_indices, device=device)
            
            outputs = model(
                images, tokens, is_closed, 
                question_types=q_type_tensor
            )
            
            # Get Predictions
            batch_preds_idx = torch.zeros_like(targets)
            
            # Closed
            if is_closed.any():
                closed_preds = outputs['closed_logits'][is_closed].argmax(dim=-1)
                batch_preds_idx[is_closed] = closed_preds
            
            # Open
            if (~is_closed).any():
                open_logits = outputs['open_logits'][~is_closed]
                open_preds = open_logits.argmax(dim=-1)
                batch_preds_idx[~is_closed] = open_preds
            
            # Process Batch
            for i in range(len(images)):
                pred_idx = batch_preds_idx[i].item()
                tgt_idx = targets[i].item()
                is_c = is_closed[i].item()
                q_text = raw_questions[i]
                img_name = image_names[i]
                
                pred_text = idx_to_ans.get(pred_idx, 'unknown')
                tgt_text = idx_to_ans.get(tgt_idx, 'unknown')
                
                # Check correctness
                is_correct = compute_exact_match(pred_text, tgt_text)
                
                # Stats
                total_samples += 1
                if is_correct: correct_overall += 1
                
                if is_c:
                    total_closed += 1
                    if is_correct: correct_closed += 1
                else:
                    total_open += 1
                    if is_correct: correct_open += 1
                    all_preds_text.append(pred_text)
                    all_targets_text.append(tgt_text)
                
                # Record result
                results.append({
                    'image': img_name,
                    'question': q_text,
                    'type': 'CLOSED' if is_c else 'OPEN',
                    'prediction': pred_text,
                    'target': tgt_text,
                    'correct': is_correct
                })

    # 6. Metrics Calculation
    acc_closed = correct_closed / total_closed if total_closed > 0 else 0
    acc_open = correct_open / total_open if total_open > 0 else 0
    acc_overall = correct_overall / total_samples if total_samples > 0 else 0
    
    bert_score = bert_evaluator.compute(all_preds_text, all_targets_text) if all_preds_text else 0.0
    
    # 7. Print Report
    print("\n" + "="*30)
    print("       TEST RESULTS       ")
    print("="*30)
    print(f"Overall Accuracy: {acc_overall:.2%}")
    print(f"Closed Accuracy:  {acc_closed:.2%}")
    print(f"Open Accuracy:    {acc_open:.2%}")
    print(f"BERTScore (Open): {bert_score:.4f}")
    print("="*30 + "\n")
    
    logger.info(f"Test Finished. Overall: {acc_overall:.2%}, Closed: {acc_closed:.2%}, Open: {acc_open:.2%}, BERT: {bert_score:.4f}")
    
    # 8. Save Detailed Results
    df = pd.DataFrame(results)
    save_path = os.path.join(cfg.training.log_dir, "test_predictions.csv")
    df.to_csv(save_path, index=False)
    print(f"Detailed predictions saved to {save_path}")

if __name__ == "__main__":
    test()
