import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_data(log_dir, tag='average_episode_rewards', max_steps=None):
    """从 TensorBoard 日志中加载数据"""
    
    # 根据 tag 名称构建正确的路径
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
    """指数移动平均平滑曲线"""
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
    """
    绘制配对的曲线对比图
    
    Args:
        run_pairs: [(baseline_dir, method_dir), ...] 配对的目录
        pair_labels: ["Baseline", "LSAM"] 两种方法的标签
        map_names: ["Map a", "Map c", "Map d"] 每对的地图名称
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 为每对使用不同的颜色
    pair_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # baseline 用虚线，method 用实线
    linestyles = ['--', '-']
    linewidths = [2, 2.5]
    
    has_data = False
    
    for pair_idx, (run_pair, map_name) in enumerate(zip(run_pairs, map_names)):
        color = pair_colors[pair_idx % len(pair_colors)]
        
        for method_idx, run_dir in enumerate(run_pair):
            # 构建 logs 目录路径
            log_dir = os.path.join(run_dir, 'logs')
            if not os.path.exists(log_dir):
                log_dir = run_dir
            
            steps, values = load_tensorboard_data(log_dir, tag, max_steps=max_steps)
            
            if steps is None or len(steps) == 0:
                print(f"Skipping {run_dir}: no data found")
                continue
            
            has_data = True
            
            # 构建标签：π₁/π₁' 格式
            subscript = ['_a','_c', '_d', '₅', '₆'][pair_idx]
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
                # 只给实线（你的方法）添加填充
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
    
    # 自定义图例，分两列显示
    ax.legend(loc='lower right', fontsize=20, ncol=1)
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 基础路径
    base_path = "../results/MyEnv/MyEnv/mappo/check"
    
    # 配对的运行目录：(baseline, LSAM)
    run_pairs = [
        (os.path.join(base_path, "run31"), os.path.join(base_path, "run32")),  # Map a
        (os.path.join(base_path, "run1"), os.path.join(base_path, "run19")),   # Map c
        (os.path.join(base_path, "run27"), os.path.join(base_path, "run26")),  # Map d
        # (os.path.join(base_path, "run38"), os.path.join(base_path, "run39")), # Map b
    ]
    
    # 方法标签
    pair_labels = ["baseline", "LSAM"]
    
    # 地图名称
    map_names = ["Map a", "Map c", "Map d"]  # , "Map b"
    
    # 绘制配对对比图
    plot_paired_runs(
        run_pairs=run_pairs,
        pair_labels=pair_labels,
        map_names=map_names,
        tag='average_episode_rewards',
        smooth=True,
        smooth_weight=0.8,
        save_path="reward_comparison_paired.png",
        max_steps=1e6
    )