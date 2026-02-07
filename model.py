"""
Enhanced BTIA-AD Net Model
Medical VQA with BioMedCLIP and Dual-Head Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import open_clip


class BioMedCLIPEncoder(nn.Module):
    """BioMedCLIP Vision and Text Encoder"""
    
    def __init__(self, model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__()
        
        # Load BioMedCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Get actual dimensions from model
        # BioMedCLIP typically uses 512D output
        self.vision_dim = self.model.visual.output_dim if hasattr(self.model.visual, 'output_dim') else 512
        self.text_dim = self.model.text.output_dim if hasattr(self.model.text, 'output_dim') else 512
        
        print(f"BioMedCLIP dimensions - Vision: {self.vision_dim}, Text: {self.text_dim}")
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to features [B, dim]"""
        return self.model.encode_image(image)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Encode text to features [B, dim]"""
        return self.model.encode_text(text_tokens)
    
    def get_image_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both patch features and global features"""
        # Get visual features from trunk
        x = self.model.visual.trunk.patch_embed(image)
        x = self.model.visual.trunk._pos_embed(x)
        
        # Pass through transformer blocks and collect multi-scale
        multi_scale = []
        for i, block in enumerate(self.model.visual.trunk.blocks):
            x = block(x)
            if i in [3, 6, 9, 11]:  # Collect at different depths
                multi_scale.append(x)
        
        x = self.model.visual.trunk.norm(x)
        
        # Global feature (CLS token or pooled)
        global_feat = x[:, 0]  # CLS token
        
        return x, global_feat, multi_scale
    
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        """Forward pass for both modalities"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features


class SemanticAnswerDistillation(nn.Module):
    """Semantic-aware Answer Distillation Network"""
    
    def __init__(self, dim: int = 768, k: int = 10, num_answers: int = 500):
        super().__init__()
        self.k = k
        self.dim = dim
        
        # Learnable answer embeddings (will be initialized from text encoder)
        self.answer_embeddings = nn.Parameter(torch.randn(num_answers, dim))
        
        # Visual-guided question attention
        self.visual_question_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Similarity projection
        self.similarity_proj = nn.Linear(dim, dim)
        
    def init_answer_embeddings(self, embeddings: torch.Tensor):
        """Initialize answer embeddings from pre-computed text embeddings"""
        with torch.no_grad():
            self.answer_embeddings.data = embeddings.clone()
    
    def forward(
        self, 
        visual_feat: torch.Tensor, 
        text_feat: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feat: [B, 768] global visual features
            text_feat: [B, 768] question features
            candidate_mask: [B, num_answers] optional mask to filter candidates
        Returns:
            topk_scores: [B, K] scores for top-k answers
            topk_indices: [B, K] indices of top-k answers
            topk_embeddings: [B, K, 768] embeddings of top-k answers
        """
        B = visual_feat.size(0)
        
        # Visual-guided question attention (single query-key-value)
        visual_feat = visual_feat.unsqueeze(1)  # [B, 1, 768]
        text_feat = text_feat.unsqueeze(1)  # [B, 1, 768]
        
        attended_question, _ = self.visual_question_attention(
            text_feat, visual_feat, visual_feat
        )
        attended_question = attended_question.squeeze(1)  # [B, 768]
        
        # Fuse visual and question
        fused = self.fusion_proj(torch.cat([visual_feat.squeeze(1), attended_question], dim=-1))
        
        # Compute similarity with answer embeddings
        fused_proj = self.similarity_proj(fused)  # [B, 768]
        answer_norm = F.normalize(self.answer_embeddings, dim=-1)  # [num_ans, 768]
        fused_norm = F.normalize(fused_proj, dim=-1)  # [B, 768]
        
        similarity = torch.matmul(fused_norm, answer_norm.T)  # [B, num_answers]
        
        # Apply candidate mask if provided
        if candidate_mask is not None:
            # Soft filtering: suppress invalid answers but don't hard remove
            # candidate_mask should be 1.0 for valid, 0.01 for invalid
            similarity = similarity * candidate_mask
        
        # Top-K selection
        topk_scores, topk_indices = torch.topk(similarity, self.k, dim=-1)  # [B, K]
        
        # Get embeddings for top-k answers
        topk_embeddings = self.answer_embeddings[topk_indices]  # [B, K, 768]
        
        return topk_scores, topk_indices, topk_embeddings


class CrossModalFusion(nn.Module):
    """Cross-Modal Transformer Fusion with Answer Guidance"""
    
    def __init__(self, dim: int = 768, heads: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention for each modality
        self.visual_self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.text_self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        
        # Cross-attention
        self.visual_to_text = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.text_to_visual = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        
        # Answer-guided attention
        self.answer_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        
        # Final projection
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(
        self, 
        visual_feat: torch.Tensor, 
        text_feat: torch.Tensor,
        answer_candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, 768]
            text_feat: [B, 768]
            answer_candidates: [B, K, 768]
        Returns:
            fused: [B, 768]
        """
        # Expand to sequence format
        visual = visual_feat.unsqueeze(1)  # [B, 1, 768]
        text = text_feat.unsqueeze(1)  # [B, 1, 768]
        
        # Self-attention
        visual_self, _ = self.visual_self_attn(visual, visual, visual)
        visual = self.norm1(visual + visual_self)
        
        text_self, _ = self.text_self_attn(text, text, text)
        text = self.norm2(text + text_self)
        
        # Cross-attention (bi-directional)
        visual_cross, _ = self.visual_to_text(visual, text, text)
        text_cross, _ = self.text_to_visual(text, visual, visual)
        
        # Combine
        fused = visual_cross + text_cross
        fused = self.norm3(fused)
        
        # Answer-guided attention
        answer_guided, attn_weights = self.answer_attn(fused, answer_candidates, answer_candidates)
        fused = self.norm4(fused + answer_guided)
        
        # FFN
        fused = fused + self.ffn(fused)
        
        return self.output_proj(fused.squeeze(1)), attn_weights


class EnhancedBTIANet(nn.Module):
    """
    Enhanced BTIA-AD Net for Medical VQA
    - BioMedCLIP encoder (medical pre-trained)
    - Semantic Answer Distillation
    - Cross-Modal Fusion
    - Dual heads: Binary (closed) and Multi-class (open)
    """
    
    def __init__(
        self,
        num_open_answers: int,
        k: int = 10,
        dim: int = 768,
        fusion_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        use_dual_heads: bool = True
    ):
        super().__init__()
        
        self.num_open_answers = num_open_answers
        self.k = k
        self.use_dual_heads = use_dual_heads
        
        # BioMedCLIP Encoder
        self.encoder = BioMedCLIPEncoder()
        
        # Freeze encoder initially (will unfreeze during training)
        self._freeze_encoder()
        
        # Answer Distillation (only for open questions)
        self.answer_distillation = SemanticAnswerDistillation(
            dim=dim, k=k, num_answers=num_open_answers
        )
        
        # Question-Type Aware Filtering Pools
        # These will be populated by the trainer
        self.question_type_masks = nn.ParameterDict()
        
        # Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            dim=dim, heads=fusion_heads, layers=fusion_layers, dropout=dropout
        )
        
        # Classification Heads
        if use_dual_heads:
            # Separate heads for closed (binary) and open (multi-class)
            self.closed_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, 2)  # Binary: yes/no
            )
            self.open_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_open_answers)
            )
        else:
            # Unified head
            self.classifier = nn.Linear(dim, num_open_answers + 2)
        
        # Question type classifier (optional: predict if closed/open)
        self.type_classifier = nn.Linear(dim, 2)
        
        # Auxiliary Heads for Multi-Task Learning
        # 1. View/Orientation Head (7 classes)
        self.view_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 7) # axial, sagittal, coronal, pa, ap, lateral, other
        )
        
        # 2. Modality/Sequence Head (5 classes)
        self.modality_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 5) # ct, mri, x-ray, ultrasound, other
        )

    def set_question_type_masks(self, idx_to_answer: Dict[int, str]):
        """Create masks for question-type answer filtering"""
        # Define pools
        pools = {
            'view': ['axial', 'sagittal', 'coronal', 'pa', 'ap', 'lateral', 'oblique', 'frontal'],
            'sequence': ['t1', 't2', 'flair', 'dwi', 'adc', 'gre', 'weighted'],
            'modality': ['ct', 'mri', 'x-ray', 'xray', 'ultrasound', 'pet', 'mammography'],
            'location': ['brain', 'chest', 'abdomen', 'pelvis', 'lung', 'liver', 'kidney', 'spine', 'head']
        }
        
        # Map type name to index
        self.type_to_idx = {name: i+1 for i, name in enumerate(pools.keys())} # 0 is general
        self.type_to_idx['general'] = 0
        
        # Create tensor: [num_types+1, num_answers]
        # Index 0 is "general" (all 1s)
        num_types = len(pools) + 1
        device = next(self.parameters()).device
        mask_tensor = torch.ones(num_types, self.num_open_answers, device=device)
        
        for name, idx in self.type_to_idx.items():
            if name == 'general': continue
            
            keywords = pools[name]
            # Start initialized to 0.1 (soft mask) instead of 0
            # This allows other answers to still be selected if confidence is high, but penalizes them
            current_mask = torch.ones(self.num_open_answers) * 0.01 
            
            count = 0
            for ans_idx, ans in idx_to_answer.items():
                if ans_idx >= self.num_open_answers: continue
                ans_lower = ans.lower()
                if any(k in ans_lower for k in keywords):
                    current_mask[ans_idx] = 1.0
                    count += 1
            
            if count > 0:
                mask_tensor[idx] = current_mask
                print(f"Created {name} mask with {count} answers")
            else:
                print(f"Warning: No answers found for {name}")
                
        self.register_buffer('question_type_masks_tensor', mask_tensor)
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, unfreeze_vision: bool = True, unfreeze_text: bool = True):
        """Unfreeze encoder parameters"""
        if unfreeze_vision:
            for param in self.encoder.model.visual.parameters():
                param.requires_grad = True
        if unfreeze_text:
            for name, param in self.encoder.model.named_parameters():
                if 'visual' not in name:
                    param.requires_grad = True
    
    def partial_unfreeze_vision(self, num_blocks: int = 4):
        """Unfreeze last N vision transformer blocks"""
        total_blocks = len(self.encoder.model.visual.trunk.blocks)
        for i, block in enumerate(self.encoder.model.visual.trunk.blocks):
            if i >= total_blocks - num_blocks:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(
        self,
        image: torch.Tensor,
        question_tokens: torch.Tensor,
        is_closed: Optional[torch.Tensor] = None,
        question_types: Optional[torch.Tensor] = None, # [B] indices
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            image: [B, 3, 224, 224]
            question_tokens: [B, max_len]
            is_closed: [B] boolean tensor indicating closed questions
            question_types: [B] integer tensor for question type indices (0=general)
            return_attention: whether to return attention weights
            
        Returns:
            Dictionary with logits and other outputs
        """
        B = image.size(0)
        
        # Encode image and question
        image_feat = self.encoder.encode_image(image)  # [B, 768]
        text_feat = self.encoder.encode_text(question_tokens)  # [B, 768]
        
        # Normalize features
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # Answer Distillation
        # 1. Get raw similarities first (need to expose this from SAD or modify it)
        # Actually SAD does fusion then similarity.
        # We can implement mask inside SAD or apply it to the similarity matrix inside SAD.
        # But SAD is a submodule.
        # Let's modify SAD call or implement masking logic here if SAD returns full similarity? 
        # SAD currently returns topk directly.
        # Let's trust SAD provides good candidates, or pass the mask TO SAD?
        # Simpler: Pass mask to SAD.
        
        # Prepare mask [B, num_answers]
        if question_types is not None and hasattr(self, 'question_type_masks_tensor'):
            # Gather masks: [B, num_answers]
            type_masks = self.question_type_masks_tensor[question_types]
        else:
            type_masks = None
            
        # We need to modify SAD to accept mask. 
        # For now, let's assume SAD is unmodified and we filter POST-hoc? 
        # No, we want distillation to pick relevant candidates.
        # So we must modify SAD.forward to accept `valid_mask`
        
        # Let's just modify SAD in this same file (it's above). 
        # Or, we can do a trick: If we can't modify SAD signature easily in this edit,
        # we can interpret the SAD output.
        # But wait, looking at SAD code (lines 65-136), it takes visual_feat, text_feat.
        # I can pass `mask` if I update SAD.forward.
        
        # Since I can't update SAD.forward in this specific replacement (it's outside range),
        # I'll update SAD.forward in a separate call or encompass it here if ranges allow.
        # The ranges are far apart (SAD is lines 65-100, BTIANet is 217+).
        # I'll just pass it and hope python doesn't crash? No.
        
        # Alternative: Don't pass mask to SAD, but use it to re-weight the fusion/classification?
        # No, SAD's job is to select candidates. If it selects "axial" for "T2", fusion sees wrong info.
        
        # I MUST update SAD.forward.
        # I will update BTIANet forward here, and assume SAD.forward is updated in next step.
        
        topk_scores, topk_indices, topk_embeddings = self.answer_distillation(
            image_feat, text_feat, candidate_mask=type_masks
        )
        
        # Cross-Modal Fusion with answer guidance
        fused, attn_weights = self.fusion(image_feat, text_feat, topk_embeddings)
        
        # Question type prediction
        type_logits = self.type_classifier(fused)
        
        # Auxiliary Head Predictions
        view_logits = self.view_head(fused)
        modality_logits = self.modality_head(fused)
        
        # Classification
        if self.use_dual_heads:
            closed_logits = self.closed_head(fused)  # [B, 2]
            open_logits = self.open_head(fused)  # [B, num_open_answers]
            
            outputs = {
                'closed_logits': closed_logits,
                'open_logits': open_logits,
                'type_logits': type_logits,
                'view_logits': view_logits,
                'modality_logits': modality_logits,
                'topk_scores': topk_scores,
                'topk_indices': topk_indices,
                'fused_features': fused
            }
        else:
            logits = self.classifier(fused)
            outputs = {
                'logits': logits,
                'type_logits': type_logits,
                'view_logits': view_logits,
                'modality_logits': modality_logits,
                'topk_scores': topk_scores,
                'topk_indices': topk_indices,
                'fused_features': fused
            }
        
        if return_attention:
            outputs['attention_weights'] = attn_weights
        
        return outputs
    
    def predict(
        self,
        image: torch.Tensor,
        question_tokens: torch.Tensor,
        is_closed: Optional[torch.Tensor] = None,
        question_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, question_tokens, is_closed, question_types)
            
            if self.use_dual_heads and is_closed is not None:
                predictions = torch.zeros(image.size(0), dtype=torch.long, device=image.device)
                confidences = torch.zeros(image.size(0), device=image.device)
                
                # Closed
                closed_mask = is_closed
                if closed_mask.any():
                    closed_probs = F.softmax(outputs['closed_logits'][closed_mask], dim=-1)
                    predictions[closed_mask] = closed_probs.argmax(dim=-1)
                    confidences[closed_mask] = closed_probs.max(dim=-1)[0]
                
                # Open
                open_mask = ~is_closed
                if open_mask.any():
                    open_probs = F.softmax(outputs['open_logits'][open_mask], dim=-1)
                    predictions[open_mask] = open_probs.argmax(dim=-1)
                    confidences[open_mask] = open_probs.max(dim=-1)[0]
                
                return predictions, confidences
            else:
                logits = outputs.get('logits', outputs['open_logits'])
                probs = F.softmax(logits, dim=-1)
                return probs.argmax(dim=-1), probs.max(dim=-1)[0]


def build_model(config, num_open_answers: int) -> EnhancedBTIANet:
    """Build model from config"""
    model = EnhancedBTIANet(
        num_open_answers=num_open_answers,
        k=config.model.top_k,
        dim=config.model.fusion_dim,
        fusion_heads=config.model.fusion_heads,
        fusion_layers=config.model.fusion_layers,
        dropout=config.model.dropout,
        use_dual_heads=config.model.use_dual_heads
    )
    return model


if __name__ == "__main__":
    # Test model
    from config import get_config
    
    config = get_config()
    model = build_model(config, num_open_answers=500)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_tokens = torch.randint(0, 1000, (2, 77))
    dummy_closed = torch.tensor([True, False])
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_image, dummy_tokens, dummy_closed)
    
    print(f"Closed logits shape: {outputs['closed_logits'].shape}")
    print(f"Open logits shape: {outputs['open_logits'].shape}")
    print(f"TopK indices shape: {outputs['topk_indices'].shape}")
