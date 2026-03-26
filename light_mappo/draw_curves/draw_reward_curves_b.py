import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_data(log_dir, tag='average_episode_rewards', max_steps=None):

    specific_log_dir = os.path.join(log_dir, tag, tag)
    
    if os.path.exists(specific_log_dir):
        actual_log_dir = specific_log_dir
    else:
        actual_log_dir = log_dir
    
    print(f"Loading from: {actual_log_dir}")
    
    ea = event_accumulator.EventAccumulator(actual_log_dir)
    ea.Reload()
    
    print(f"Available tags: {ea.Tags()['scalars']}")
    
    if not ea.Tags()['scalars']:
        print(f"No scalar tags found in {actual_log_dir}")
        return None, None
    
    available_tag = ea.Tags()['scalars'][0] if ea.Tags()['scalars'] else None
    
    if available_tag is None:
        return None, None
    
    print(f"Using tag: {available_tag}")
    
    events = ea.Scalars(available_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    steps = np.array(steps)
    values = np.array(values)
    
    if max_steps is not None:
        mask = steps <= max_steps
        steps = steps[mask]
        values = values[mask]
    
    return steps, values

def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_paired_runs(run_pairs, pair_labels, map_names,
                     tag='average_episode_rewards', 
                     smooth=True, smooth_weight=0.9, 
                     save_path=None, max_steps=None):

    fig, ax = plt.subplots(figsize=(8, 7))
    

    pair_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    

    linestyles = ['--', '-']
    linewidths = [2, 2.5]
    
    has_data = False
    
    for pair_idx, (run_pair, map_name) in enumerate(zip(run_pairs, map_names)):
        color = pair_colors[pair_idx % len(pair_colors)]
        
        for method_idx, run_dir in enumerate(run_pair):

            log_dir = os.path.join(run_dir, 'logs')
            if not os.path.exists(log_dir):
                log_dir = run_dir
            
            steps, values = load_tensorboard_data(log_dir, tag, max_steps=max_steps)
            
            if steps is None or len(steps) == 0:
                print(f"Skipping {run_dir}: no data found")
                continue
            
            has_data = True
            

            subscript = ['_b'][pair_idx]
            if method_idx == 0:
                label = f"π{subscript}: {map_name} ({pair_labels[0]})"
            else:
                label = f"π{subscript}': {map_name} ({pair_labels[1]})"
            
            if smooth:
                smoothed_values = smooth_curve(values, smooth_weight)
                ax.plot(steps, smoothed_values, 
                       label=label, 
                       color=color, 
                       linestyle=linestyles[method_idx],
                       linewidth=linewidths[method_idx])

                if method_idx == 1:
                    ax.fill_between(steps, values, smoothed_values, 
                                   alpha=0.1, color=color)
            else:
                ax.plot(steps, values, 
                       label=label, 
                       color=color, 
                       linestyle=linestyles[method_idx],
                       linewidth=linewidths[method_idx])
    
    if not has_data:
        print("No data to plot!")
        return
    
    ax.set_xlabel('Training Steps', fontsize=17)
    ax.set_ylabel('Average Episode Rewards', fontsize=17)
    # ax.set_title('Training Reward Comparison: Baseline vs LSAM', fontsize=14)
    
    ax.tick_params(axis='both', labelsize=17)
    ax.xaxis.get_offset_text().set_fontsize(17)
    
    ax.legend(loc='upper left', fontsize=24, ncol=1)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    base_path = "../results/MyEnv/MyEnv/mappo/check"
    
    run_pairs = [
        # (os.path.join(base_path, "run31"), os.path.join(base_path, "run32")),  # Map a
        # (os.path.join(base_path, "run1"), os.path.join(base_path, "run19")),   # Map c
        # (os.path.join(base_path, "run27"), os.path.join(base_path, "run26")),  # Map d
        (os.path.join(base_path, "run38"), os.path.join(base_path, "run39")), # Map b
    ]
    
    pair_labels = ["Baseline", "LSAM"]
    
    map_names = ["Map b"]  # , "Map b"
    
    plot_paired_runs(
        run_pairs=run_pairs,
        pair_labels=pair_labels,
        map_names=map_names,
        tag='average_episode_rewards',
        smooth=True,
        smooth_weight=0.8,
        save_path="reward_comparison_paired.png",
        max_steps=2e6
    )