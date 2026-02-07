"""
Inference and Evaluation for Enhanced BTIA-AD Net
"""

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import open_clip

from config import get_config
from model import EnhancedBTIANet


class MedVQAInference:
    """Inference engine for Medical VQA"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get answer mappings
        self.answer_to_idx = checkpoint['answer_to_idx']
        self.idx_to_answer = checkpoint['idx_to_answer']
        self.closed_answers = {0: 'yes', 1: 'no'}
        
        # Build model
        num_answers = len(self.answer_to_idx)
        self.model = EnhancedBTIANet(num_open_answers=num_answers)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get tokenizer and preprocessor
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        
        print(f"Model loaded! {num_answers} open answer classes")
    
    def predict(
        self,
        image_path: str,
        question: str,
        is_closed: Optional[bool] = None
    ) -> Dict:
        """
        Predict answer for a single image-question pair
        
        Args:
            image_path: Path to the medical image
            question: Question about the image
            is_closed: Whether it's a yes/no question (auto-detected if None)
            
        Returns:
            Dictionary with prediction, confidence, and top-k candidates
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize question
        question_tokens = self.tokenizer([question]).to(self.device)
        
        # Auto-detect question type if not provided
        if is_closed is None:
            question_lower = question.lower()
            is_closed = any(word in question_lower for word in [
                'is there', 'are there', 'is this', 'is the', 'does',
                'do you', 'can you', 'is it', 'are they'
            ])
        
        is_closed_tensor = torch.tensor([is_closed], device=self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor, question_tokens, is_closed_tensor, return_attention=True)
        
        # Get prediction
        if is_closed:
            probs = F.softmax(outputs['closed_logits'], dim=-1)[0]
            pred_idx = probs.argmax().item()
            prediction = self.closed_answers[pred_idx]
            confidence = probs.max().item()
        else:
            probs = F.softmax(outputs['open_logits'], dim=-1)[0]
            pred_idx = probs.argmax().item()
            prediction = self.idx_to_answer.get(pred_idx, 'unknown')
            confidence = probs.max().item()
        
        # Get top-k candidates from distillation
        topk_indices = outputs['topk_indices'][0].cpu().numpy()
        topk_scores = outputs['topk_scores'][0].cpu().numpy()
        topk_answers = [self.idx_to_answer.get(i, 'unknown') for i in topk_indices]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'is_closed': is_closed,
            'top_k_candidates': list(zip(topk_answers, topk_scores.tolist())),
            'attention_weights': outputs.get('attention_weights', None)
        }
    
    def predict_batch(
        self,
        image_paths: List[str],
        questions: List[str],
        is_closed_list: Optional[List[bool]] = None
    ) -> List[Dict]:
        """Predict for a batch of image-question pairs"""
        results = []
        for i, (img_path, question) in enumerate(zip(image_paths, questions)):
            is_closed = is_closed_list[i] if is_closed_list else None
            result = self.predict(img_path, question, is_closed)
            results.append(result)
        return results
    
    def visualize_attention(
        self,
        image_path: str,
        question: str,
        save_path: Optional[str] = None
    ):
        """Visualize attention map overlaid on image"""
        # Get prediction with attention
        result = self.predict(image_path, question)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get attention weights
        attn = result.get('attention_weights')
        if attn is not None:
            attn = attn[0].mean(dim=0).cpu().numpy()  # Average over heads
            
            # Reshape to spatial dimensions (assuming 14x14 for ViT-B/16)
            attn_map = attn.mean(axis=0)  # [K] -> scalar per candidate
            # For visualization, we'll show the answer-candidate attention
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title(f"Q: {question}")
        axes[0].axis('off')
        
        # Prediction info
        axes[1].text(0.5, 0.7, f"Prediction: {result['prediction']}", 
                    fontsize=16, ha='center', transform=axes[1].transAxes)
        axes[1].text(0.5, 0.5, f"Confidence: {result['confidence']:.2%}", 
                    fontsize=14, ha='center', transform=axes[1].transAxes)
        axes[1].text(0.5, 0.3, f"Type: {'Closed (Yes/No)' if result['is_closed'] else 'Open'}", 
                    fontsize=12, ha='center', transform=axes[1].transAxes)
        
        # Top candidates
        candidates_text = "Top Candidates:\n"
        for ans, score in result['top_k_candidates'][:5]:
            candidates_text += f"  {ans}: {score:.3f}\n"
        axes[1].text(0.5, 0.1, candidates_text, fontsize=10, ha='center', 
                    transform=axes[1].transAxes, va='top')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        return result


def evaluate_model(checkpoint_path: str, test_json: str, images_dir: str):
    """Evaluate model on test set"""
    # Load inference engine
    engine = MedVQAInference(checkpoint_path)
    
    # Load test data
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    correct_closed = 0
    correct_open = 0
    total_closed = 0
    total_open = 0
    
    print(f"Evaluating on {len(test_data)} samples...")
    
    for item in test_data:
        image_path = os.path.join(images_dir, item['image'])
        question = item['question']
        gt_answer = item['answer'].lower().strip()
        
        is_closed = gt_answer in ['yes', 'no']
        
        result = engine.predict(image_path, question, is_closed)
        pred = result['prediction'].lower().strip()
        
        if is_closed:
            total_closed += 1
            if pred == gt_answer:
                correct_closed += 1
        else:
            total_open += 1
            if pred == gt_answer:
                correct_open += 1
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Closed-ended Accuracy: {correct_closed}/{total_closed} = {100*correct_closed/total_closed:.2f}%")
    print(f"Open-ended Accuracy: {correct_open}/{total_open} = {100*correct_open/total_open:.2f}%")
    print(f"Overall Accuracy: {correct_closed+correct_open}/{total_closed+total_open} = {100*(correct_closed+correct_open)/(total_closed+total_open):.2f}%")
    print("="*50)
    
    return {
        'closed_acc': correct_closed / total_closed,
        'open_acc': correct_open / total_open,
        'overall_acc': (correct_closed + correct_open) / (total_closed + total_open)
    }


# Example usage
if __name__ == "__main__":
    # Example inference
    print("="*60)
    print("Enhanced BTIA-AD Net - Inference Example")
    print("="*60)
    
    # Check if model exists
    checkpoint_path = "checkpoints/best.pth"
    
    if os.path.exists(checkpoint_path):
        # Load inference engine
        engine = MedVQAInference(checkpoint_path)
        
        # Example prediction
        example_image = "data/vqa_rad/images/test_0001.jpg"
        example_question = "Is there a fracture visible?"
        
        if os.path.exists(example_image):
            result = engine.predict(example_image, example_question)
            print(f"\nQuestion: {example_question}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Type: {'Closed' if result['is_closed'] else 'Open'}")
            
            # Visualize
            engine.visualize_attention(example_image, example_question, 'prediction_viz.png')
        else:
            print(f"Example image not found: {example_image}")
    else:
        print(f"No trained model found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
