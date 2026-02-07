
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_styled_box(ax, x, y, width, height, text, color, subtext=None):
    # Shadow effect
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), width, height, boxstyle="round,pad=0.2", 
                                  linewidth=0, facecolor='#dddddd', zorder=1)
    ax.add_patch(shadow)
    
    # Main Box
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                  linewidth=1.5, edgecolor='#333333', facecolor=color, zorder=2)
    ax.add_patch(rect)
    
    # Text
    ax.text(x + width/2, y + height/2 + (0.1 if subtext else 0), text, 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50', zorder=3)
    
    if subtext:
        ax.text(x + width/2, y + height/2 - 0.2, subtext, 
                ha='center', va='center', fontsize=8, style='italic', color='#555555', zorder=3)
        
    return {
        'top': (x + width/2, y + height + 0.2), 
        'bottom': (x + width/2, y - 0.2),
        'left': (x - 0.2, y + height/2),
        'right': (x + width + 0.2, y + height/2),
        'center': (x + width/2, y + height/2)
    }

def draw_section(ax, x, y, width, height, title, color):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", 
                                  linewidth=1, edgecolor=color, facecolor=color, alpha=0.1, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.2, y + height - 0.3, title, fontsize=12, fontweight='bold', color=color, ha='left', zorder=1)

def connect(ax, start, end, style="arc3,rad=0", color='#555555'):
    ax.annotate("", xy=end, xytext=start, zorder=0,
                arrowprops=dict(arrowstyle="simple,head_width=0.6,head_length=0.6", 
                                connectionstyle=style, color=color, alpha=0.8))

def generate_diagram():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # --- Layout Constants ---
    W_BOX = 2.5
    H_BOX = 0.8
    CENTER_X = 6
    LEFT_X = 3
    RIGHT_X = 9
    
    # Palettes
    C_DATA = '#e3f2fd'  # Light Blue
    C_MODEL = '#e8f5e9' # Light Green
    C_CORE = '#fff3e0'  # Light Orange
    C_OUT = '#fce4ec'   # Light Pink
    C_ACCENT = '#1565c0' # Dark Blue Text
    
    # ==========================================
    # 1. Data Layer
    # ==========================================
    draw_section(ax, 0.5, 13.5, 11, 4, "DATA PIPELINE", '#1976d2')
    
    # Sources
    src1 = draw_styled_box(ax, LEFT_X-1, 16, W_BOX, H_BOX, "VQA-RAD", C_DATA, "(Radiology)")
    src2 = draw_styled_box(ax, RIGHT_X-1.5, 16, W_BOX, H_BOX, "PathVQA", C_DATA, "(Pathology)")
    
    # Merge
    merge = draw_styled_box(ax, CENTER_X - W_BOX/2, 14.5, W_BOX, H_BOX, "Merge & Split", '#bbdefb', "80/20 Random")
    
    connect(ax, src1['bottom'], merge['top'], "arc3,rad=-0.1")
    connect(ax, src2['bottom'], merge['top'], "arc3,rad=0.1")
    
    # ==========================================
    # 2. Representation Layer
    # ==========================================
    draw_section(ax, 0.5, 8.5, 11, 4.5, "REPRESENTATION LAYER", '#388e3c')
    
    # Encoders
    vis_enc = draw_styled_box(ax, LEFT_X-1, 11, W_BOX, H_BOX, "Vision Encoder", C_MODEL, "BioMedCLIP ViT")
    txt_enc = draw_styled_box(ax, RIGHT_X-1.5, 11, W_BOX, H_BOX, "Text Encoder", C_MODEL, "PubMedBERT")
    
    # Augmentation indicator arrow
    ax.text(LEFT_X-1 + W_BOX/2, 12.8, "Augmentation\n(No Flip)", ha='center', fontsize=8, color='#555')
    connect(ax, merge['bottom'], vis_enc['top'], "arc3,rad=-0.1")
    connect(ax, merge['bottom'], txt_enc['top'], "arc3,rad=0.1")
    
    # Distillation Feature
    sad = draw_styled_box(ax, CENTER_X - W_BOX/2, 9.5, W_BOX, H_BOX, "Semantic\nDistillation", '#fff9c4', "Top-K Selection")
    
    connect(ax, vis_enc['right'], sad['left'], "arc3,rad=0")
    connect(ax, txt_enc['left'], sad['right'], "arc3,rad=0")
    
    # ==========================================
    # 3. Fusion & Core
    # ==========================================
    draw_section(ax, 0.5, 4.5, 11, 3.5, "CORE LOGIC", '#fbc02d')
    
    fusion = draw_styled_box(ax, CENTER_X - W_BOX/2, 6, W_BOX, H_BOX, "Cross-Modal\nFusion", C_CORE, "Transformer")
    
    # Connections to Fusion
    connect(ax, sad['bottom'], fusion['top'], "arc3,rad=0")
    # Residual connections from encoders
    connect(ax, vis_enc['bottom'], fusion['left'], "arc3,rad=-0.3")
    connect(ax, txt_enc['bottom'], fusion['right'], "arc3,rad=0.3")
    
    # ==========================================
    # 4. Output Layer
    # ==========================================
    draw_section(ax, 0.5, 0.5, 11, 3.5, "OUTPUT & PREDICTION", '#c2185b')
    
    # Heads
    head_c = draw_styled_box(ax, LEFT_X, 2.5, W_BOX, H_BOX, "Closed Head", C_OUT, "Binary (Yes/No)")
    head_o = draw_styled_box(ax, RIGHT_X-2.5, 2.5, W_BOX, H_BOX, "Open Head", C_OUT, "Multi-Class")
    
    connect(ax, fusion['bottom'], head_c['top'], "arc3,rad=-0.1")
    connect(ax, fusion['bottom'], head_o['top'], "arc3,rad=0.1")
    
    # Final
    final = draw_styled_box(ax, CENTER_X - W_BOX/2, 0.8, W_BOX, H_BOX, "FINAL\nPREDICTION", '#ff8a80')
    
    connect(ax, head_c['bottom'], final['top'], "arc3,rad=0.1")
    connect(ax, head_o['bottom'], final['top'], "arc3,rad=-0.1")
    
    # Title
    plt.suptitle("Enhanced BTIA-AD Net Architecture", fontsize=20, fontweight='bold', color='#1a1a1a', y=0.98)
    
    save_path = "workflow_diagram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Hierarchical diagram saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_diagram()
