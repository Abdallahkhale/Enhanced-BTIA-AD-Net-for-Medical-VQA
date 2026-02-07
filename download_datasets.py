import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import shutil

def download_slake(data_dir):
    """
    Download SLAKE dataset (English subset)
    Source: https://huggingface.co/datasets/BoKelvin/SLAKE
    """
    print("\n" + "="*40)
    print("Downloading SLAKE Dataset...")
    print("="*40)
    
    save_dir = os.path.join(data_dir, "slake")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        # Load dataset
        # 'English' split usually
        dataset = load_dataset("BoKelvin/SLAKE", split="train+test") # Check splits
        # If 'BoKelvin/SLAKE' fails, we might need a specific config
        # Actually, let's try a known working one or handle error
    except Exception as e:
        print(f"Error loading SLAKE from HuggingFace: {e}")
        print("Try visiting: https://med-vqa.com/slake/ for manual download.")
        return

    print(f"Found {len(dataset)} samples. Processing...")
    
    vqa_data = []
    
    for idx, item in enumerate(tqdm(dataset)):
        if idx == 0:
            print(f"DEBUG: Keys found: {item.keys()}")
            print(f"DEBUG: Sample item: {item}")
            
        # Try to find image
        image = None
        if 'image' in item:
            image = item['image']
        elif 'img_name' in item:
            # Maybe it provides a path or URL?
            # If it's just a name, we can't get the pixel data from the text-only dataset
            pass
            
        # item['question']
        # item['answer']
        # item['q_lang'] -> 'en' or 'zh'
        
        if item.get('q_lang', 'en') != 'en':
            continue
            
        # Save Image
        if not isinstance(image, Image.Image):
             # If no image object, maybe skip or warn
             if idx == 0: print("WARNING: No PIL Image found in item.")
             continue 
            
        img_name = f"slake_{idx}.jpg"
        img_path = os.path.join(img_dir, img_name)
        image.save(img_path)
        
        # Save QA
        vqa_data.append({
            "image": img_name,
            "question": item['question'],
            "answer": str(item['answer']),
            "answer_type": item.get('answer_type', 'OPEN'),
            "dataset": "slake"
        })
        
    # Save JSON
    with open(os.path.join(save_dir, "slake_vqa.json"), 'w') as f:
        json.dump(vqa_data, f, indent=2)
        
    print(f"Successfully saved {len(vqa_data)} English samples to {save_dir}")

def download_pathvqa(data_dir):
    """
    Download PathVQA dataset
    Source: https://huggingface.co/datasets/flaviagiammarino/path-vqa
    """
    print("\n" + "="*40)
    print("Downloading PathVQA Dataset...")
    print("="*40)
    
    save_dir = os.path.join(data_dir, "pathvqa")
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        # Check splits: train, val, test
        dataset = load_dataset("flaviagiammarino/path-vqa", split="train") 
    except Exception as e:
        print(f"Error loading PathVQA: {e}")
        return

    print(f"Found {len(dataset)} training samples. Processing...")
    
    vqa_data = []
    
    for idx, item in enumerate(tqdm(dataset)):
        # item['image']
        # item['question']
        # item['answer']
        
        image = item['image']
        img_name = f"pathvqa_{idx}.jpg"
        img_path = os.path.join(img_dir, img_name)
        image.save(img_path)
        
        vqa_data.append({
            "image": img_name,
            "question": item['question'],
            "answer": item['answer'],
            "dataset": "pathvqa"
        })
        
    with open(os.path.join(save_dir, "pathvqa_train.json"), 'w') as f:
        json.dump(vqa_data, f, indent=2)
        
    print(f"Saved PathVQA to {save_dir}")

if __name__ == "__main__":
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)
    
    print("Starting Medical Dataset Download...")
    print("NOTE: This script uses HuggingFace 'datasets'. Ensure you have internet access.")
    
    download_pathvqa(base_dir) 
    # download_slake(base_dir) # SLAKE (BoKelvin) missing images in this repo

