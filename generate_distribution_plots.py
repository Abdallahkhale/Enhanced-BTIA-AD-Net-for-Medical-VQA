
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from config import get_config

def generate_plots():
    print("Loading data for analysis...")
    cfg = get_config()
    
    # Paths (Hardcoded based on dataset.py logic for robustness)
    vqa_rad_train = "data/vqa_rad/train.json"
    vqa_rad_test = "data/vqa_rad/test.json"
    pathvqa_path = "data/pathvqa/pathvqa_train.json"
    
    # Load VQA-RAD
    with open(vqa_rad_train, 'r', encoding='utf-8') as f:
        rad_data = json.load(f)
    if os.path.exists(vqa_rad_test):
        with open(vqa_rad_test, 'r', encoding='utf-8') as f:
            rad_data += json.load(f)
            
    # Load PathVQA
    path_data = []
    if os.path.exists(pathvqa_path):
        with open(pathvqa_path, 'r', encoding='utf-8') as f:
            path_data = json.load(f)
    
    total_samples = len(rad_data) + len(path_data)
    print(f"Total Samples: {total_samples}")
    full_pool = rad_data + path_data

    # --- Plot 1: Dataset Source Distribution ---
    print("Generating Dataset Source Plot...")
    sources = ['VQA-RAD', 'PathVQA']
    counts = [len(rad_data), len(path_data)]
    colors = ['#4e79a7', '#f28e2b']
    
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=sources, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.1, 0))
    plt.title('Dataset Source Composition', fontsize=14, fontweight='bold')
    
    save_path1 = "dataset_source_composition.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path1}")
    
    # --- Plot 2: Train/Test Split ---
    print("Generating Train/Test Split Plot...")
    train_set, test_set = train_test_split(full_pool, test_size=0.2, random_state=cfg.training.seed)
    
    split_labels = ['Training Set\n(80%)', 'Test/Val Set\n(20%)']
    split_counts = [len(train_set), len(test_set)]
    split_colors = ['#59a14f', '#e15759']
    
    plt.figure(figsize=(8, 6))
    plt.pie(split_counts, labels=split_labels, autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * total_samples / 100, p), 
            colors=split_colors, startangle=140)
    plt.title('Train / Test Split', fontsize=14, fontweight='bold')
    
    save_path2 = "train_test_split.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path2}")
    
    # --- Plot 3: Question Type Distribution ---
    print("Generating Question Type Plot...")
    def is_closed(item):
        return item['answer'].lower().strip() in ['yes', 'no']
        
    closed_count = sum(1 for x in full_pool if is_closed(x))
    open_count = len(full_pool) - closed_count
    
    type_labels = ['Closed (Yes/No)', 'Open (Other)']
    type_counts = [closed_count, open_count]
    type_colors = ['#76b7b2', '#edc948']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(type_labels, type_counts, color=type_colors)
    plt.title('Question Type Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}\n({height/total_samples*100:.1f}%)',
                 ha='center', va='bottom')
    
    save_path3 = "question_type_distribution.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path3}")

if __name__ == "__main__":
    generate_plots()
