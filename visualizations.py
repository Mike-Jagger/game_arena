import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
STORAGE_DIR = "storage"
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metrics(game, model_name):
    filepath = os.path.join(STORAGE_DIR, f"{game}_{model_name}_metrics.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    print(f"Warning: Could not find {filepath}")
    return None

def moving_average(data, window_size=50):
    """Calculates a moving average to smooth out noisy RL training curves."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_learning_curves(q_metrics, dqn_metrics):
    """Plots the smoothed score progression of both models over time."""
    plt.figure(figsize=(10, 6))
    
    if q_metrics and 'scores_history' in q_metrics:
        q_scores = q_metrics['scores_history']
        plt.plot(moving_average(q_scores), label='Q-Learning (Smoothed)', color='blue', alpha=0.8)
        
    if dqn_metrics and 'scores_history' in dqn_metrics:
        dqn_scores = dqn_metrics['scores_history']
        plt.plot(moving_average(dqn_scores), label='DQN (Smoothed)', color='orange', alpha=0.8)

    plt.title('Snake AI Performance: Q-Learning vs. DQN', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes (Adjusted for Moving Average Window)', fontsize=12)
    plt.ylabel('Score per Episode', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, 'learning_curves.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

def plot_dqn_loss(dqn_metrics):
    """Plots the training loss curve specifically for the Deep Q-Network."""
    if not dqn_metrics or 'loss_history' not in dqn_metrics:
        return
        
    plt.figure(figsize=(10, 6))
    losses = dqn_metrics['loss_history']
    
    plt.plot(moving_average(losses, window_size=20), color='red')
    plt.title('DQN Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Mean Squared Error Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, 'dqn_loss_curve.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

def plot_final_comparison(q_metrics, dqn_metrics):
    """Creates a bar chart comparing the final average scores and absolute max scores."""
    if not q_metrics or not dqn_metrics:
        return

    labels = ['Q-Learning', 'DQN']
    avg_scores = [q_metrics.get('final_avg_score', 0), dqn_metrics.get('final_avg_score', 0)]
    max_scores = [q_metrics.get('max_score', 0), dqn_metrics.get('max_score', 0)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, avg_scores, width, label='Final Avg Score', color='teal')
    rects2 = ax.bar(x + width/2, max_scores, width, label='Max Score Achieved', color='lightseagreen')

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Final Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend()

    # Add numeric labels on top of the bars
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'final_comparison_bar.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == '__main__':
    print("Loading metrics data...")
    q_data = load_metrics("snake", "qlearning")
    dqn_data = load_metrics("snake", "dqn")
    
    if q_data or dqn_data:
        print("Generating visualizations...")
        plot_learning_curves(q_data, dqn_data)
        plot_dqn_loss(dqn_data)
        plot_final_comparison(q_data, dqn_data)
        print("\nAll visualizations generated successfully in the 'visualizations/' folder.")
    else:
        print("Error: No metrics files found. Have you trained the models yet?")