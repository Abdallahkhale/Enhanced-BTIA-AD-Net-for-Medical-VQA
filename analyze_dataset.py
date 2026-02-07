"""
VQA-RAD Dataset Analysis
Full statistical analysis of the dataset
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_dir="data/vqa_rad"):
    """Load train and test data"""
    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(os.path.join(data_dir, 'test.json'), 'r', encoding='utf-8') as f:
        test = json.load(f)
    return train, test

def analyze_dataset(data, name="Dataset"):
    """Full analysis of a dataset split"""
    print(f"\n{'='*60}")
    print(f" {name} Analysis")
    print(f"{'='*60}")
    
    # Basic counts
    total = len(data)
    print(f"\nTotal samples: {total}")
    
    # Answer analysis
    answers = [d['answer'].lower().strip() for d in data]
    answer_counts = Counter(answers)
    
    # Closed vs Open questions
    yes_count = answer_counts.get('yes', 0)
    no_count = answer_counts.get('no', 0)
    closed_count = yes_count + no_count
    open_count = total - closed_count
    
    print(f"\n--- Question Type Distribution ---")
    print(f"Closed (Yes/No): {closed_count} ({100*closed_count/total:.1f}%)")
    print(f"  - Yes: {yes_count} ({100*yes_count/total:.1f}%)")
    print(f"  - No:  {no_count} ({100*no_count/total:.1f}%)")
    print(f"Open-ended:      {open_count} ({100*open_count/total:.1f}%)")
    
    # Yes/No balance
    if closed_count > 0:
        print(f"\n--- Yes/No Balance ---")
        print(f"Yes ratio: {100*yes_count/closed_count:.1f}%")
        print(f"No ratio:  {100*no_count/closed_count:.1f}%")
    
    # Open answer analysis
    open_answers = [a for a in answers if a not in ['yes', 'no']]
    open_answer_counts = Counter(open_answers)
    
    print(f"\n--- Open Answer Statistics ---")
    print(f"Unique open answers: {len(open_answer_counts)}")
    print(f"Answers appearing once: {sum(1 for c in open_answer_counts.values() if c == 1)}")
    print(f"Answers appearing 2-5 times: {sum(1 for c in open_answer_counts.values() if 2 <= c <= 5)}")
    print(f"Answers appearing 5+ times: {sum(1 for c in open_answer_counts.values() if c > 5)}")
    
    print(f"\n--- Top 20 Open Answers ---")
    for i, (ans, count) in enumerate(open_answer_counts.most_common(20), 1):
        print(f"  {i:2d}. '{ans}': {count}")
    
    # Question analysis
    questions = [d['question'].lower() for d in data]
    
    # Question type patterns
    question_starts = Counter([q.split()[0] if q.split() else 'empty' for q in questions])
    print(f"\n--- Question Start Words ---")
    for word, count in question_starts.most_common(10):
        print(f"  '{word}': {count}")
    
    # Question length
    q_lengths = [len(q.split()) for q in questions]
    print(f"\n--- Question Length Statistics ---")
    print(f"Min: {min(q_lengths)} words")
    print(f"Max: {max(q_lengths)} words")
    print(f"Mean: {np.mean(q_lengths):.1f} words")
    print(f"Median: {np.median(q_lengths):.1f} words")
    
    return {
        'total': total,
        'closed': closed_count,
        'open': open_count,
        'yes': yes_count,
        'no': no_count,
        'unique_open_answers': len(open_answer_counts),
        'answer_counts': answer_counts,
        'open_answer_counts': open_answer_counts
    }

def create_visualizations(train_stats, test_stats, save_dir="data/vqa_rad"):
    """Create and save analysis visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('VQA-RAD Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Train vs Test split
    ax = axes[0, 0]
    splits = ['Train', 'Test']
    counts = [train_stats['total'], test_stats['total']]
    bars = ax.bar(splits, counts, color=['#2ecc71', '#3498db'])
    ax.set_title('Dataset Split')
    ax.set_ylabel('Number of Samples')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(count), ha='center', fontweight='bold')
    
    # 2. Closed vs Open (Train)
    ax = axes[0, 1]
    labels = ['Closed\n(Yes/No)', 'Open-ended']
    sizes = [train_stats['closed'], train_stats['open']]
    colors = ['#e74c3c', '#9b59b6']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Train: Question Types')
    
    # 3. Yes vs No distribution (Train)
    ax = axes[0, 2]
    labels = ['Yes', 'No']
    sizes = [train_stats['yes'], train_stats['no']]
    colors = ['#27ae60', '#c0392b']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Train: Yes/No Distribution')
    
    # 4. Top 10 Open Answers (Train)
    ax = axes[1, 0]
    top_answers = train_stats['open_answer_counts'].most_common(10)
    answers = [a[0][:15] + '...' if len(a[0]) > 15 else a[0] for a in top_answers]
    counts = [a[1] for a in top_answers]
    y_pos = np.arange(len(answers))
    ax.barh(y_pos, counts, color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(answers)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('Train: Top 10 Open Answers')
    
    # 5. Answer frequency distribution (Train)
    ax = axes[1, 1]
    freq_counts = list(train_stats['open_answer_counts'].values())
    bins = [1, 2, 3, 5, 10, 20, max(freq_counts)+1]
    ax.hist(freq_counts, bins=bins, color='#9b59b6', edgecolor='white')
    ax.set_xlabel('Answer Frequency')
    ax.set_ylabel('Number of Unique Answers')
    ax.set_title('Train: Open Answer Frequency Distribution')
    ax.set_yscale('log')
    
    # 6. Comparison Train vs Test
    ax = axes[1, 2]
    x = np.arange(3)
    width = 0.35
    train_vals = [train_stats['closed'], train_stats['open'], train_stats['unique_open_answers']]
    test_vals = [test_stats['closed'], test_stats['open'], test_stats['unique_open_answers']]
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='#2ecc71')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='#3498db')
    ax.set_ylabel('Count')
    ax.set_title('Train vs Test Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Closed', 'Open', 'Unique Answers'])
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'dataset_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    plt.show()

def main():
    """Run full dataset analysis"""
    print("\n" + "="*60)
    print(" VQA-RAD DATASET FULL ANALYSIS")
    print("="*60)
    
    # Load data
    train, test = load_data()
    
    # Analyze both splits
    train_stats = analyze_dataset(train, "TRAIN SET")
    test_stats = analyze_dataset(test, "TEST SET")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(" SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'Train':>10} {'Test':>10} {'Total':>10}")
    print("-"*55)
    print(f"{'Total Samples':<25} {train_stats['total']:>10} {test_stats['total']:>10} {train_stats['total']+test_stats['total']:>10}")
    print(f"{'Closed (Yes/No)':<25} {train_stats['closed']:>10} {test_stats['closed']:>10} {train_stats['closed']+test_stats['closed']:>10}")
    print(f"{'Open-ended':<25} {train_stats['open']:>10} {test_stats['open']:>10} {train_stats['open']+test_stats['open']:>10}")
    print(f"{'Yes answers':<25} {train_stats['yes']:>10} {test_stats['yes']:>10} {train_stats['yes']+test_stats['yes']:>10}")
    print(f"{'No answers':<25} {train_stats['no']:>10} {test_stats['no']:>10} {train_stats['no']+test_stats['no']:>10}")
    print(f"{'Unique open answers':<25} {train_stats['unique_open_answers']:>10} {test_stats['unique_open_answers']:>10} {'-':>10}")
    
    # Key insights
    print(f"\n{'='*60}")
    print(" KEY INSIGHTS")
    print(f"{'='*60}")
    print(f"""
1. CLASS IMBALANCE:
   - Closed questions dominate ({100*train_stats['closed']/train_stats['total']:.1f}% of train)
   - Yes/No ratio: {train_stats['yes']}:{train_stats['no']} = {train_stats['yes']/train_stats['no']:.2f}:1

2. OPEN ANSWER CHALLENGE:
   - {train_stats['unique_open_answers']} unique answers for {train_stats['open']} open questions
   - Many answers appear only once (long-tail distribution)
   - This makes open-ended VQA very challenging

3. TRAIN/TEST SPLIT:
   - Train: {train_stats['total']} samples ({100*train_stats['total']/(train_stats['total']+test_stats['total']):.1f}%)
   - Test: {test_stats['total']} samples ({100*test_stats['total']/(train_stats['total']+test_stats['total']):.1f}%)

4. RECOMMENDATIONS:
   - Use separate heads for closed (binary) vs open (multi-class)
   - Apply answer distillation to handle sparse open answers
   - Consider class weighting for imbalanced Yes/No
""")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(train_stats, test_stats)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
