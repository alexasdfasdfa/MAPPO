import os
import sys
import matplotlib
import shutil
matplotlib.use('Agg')  # 适用于无图形界面的服务器环境
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_event_file(directory):
    """在指定目录下递归查找第一个以 events.out.tfevents 开头的文件，返回完整路径，否则返回 None"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('events.out.tfevents'):
                return os.path.join(root, file)
    return None

def main():
    if len(sys.argv) != 2:
        print("用法: python script.py <运行编号>")
        sys.exit(1)

    dir_name = os.path.join('fig', 'train')
    if not os.path.isdir('fig'): os.mkdir('fig')
    if not os.path.isdir(dir_name): os.mkdir(dir_name) 

    run_number = sys.argv[1]
    dir_name = os.path.join(dir_name, f'{run_number}')
    if os.path.isdir(dir_name): shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    base_dir = f"results/train/run{run_number}/logs"
    if not os.path.isdir(base_dir):
        print(f"错误：目录 {base_dir} 不存在")
        sys.exit(1)

    labels = ['average_episode_rewards', 'policy_loss', 'value_loss']

    for label in labels:
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            print(f"警告：标签目录 {label_dir} 不存在，跳过 {label}")
            continue

        event_file = find_event_file(label_dir)
        if event_file is None:
            print(f"错误：在 {label_dir} 下未找到事件文件，跳过 {label}")
            continue

        print(f"\n处理 {label}，事件文件：{event_file}")
        try:
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()
        except Exception as e:
            print(f"加载事件文件失败：{e}")
            continue

        try:
            scalars = event_acc.Scalars(label)
        except KeyError:
            print(f"事件文件中没有标签 {label}")
            continue

        if not scalars:
            print(f"没有标量数据")
            continue

        # 打印数据（为避免输出过多，只显示前 10 条；如需全部打印可移除限制）
        print(f"{label} 数据 (步数, 值):")
        for i, scalar in enumerate(scalars):
            if i < 10:
                print(f"  Step: {scalar.step}, Value: {scalar.value}")
            else:
                print(f"  ... 共 {len(scalars)} 条数据")
                break

        # 提取步数和值
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o', linestyle='-', markersize=3)
        plt.xlabel('Step')
        plt.ylabel(label.replace('_', ' ').title())
        plt.title(f'Training Progress: {label} vs Step (Run {run_number})')
        plt.grid(True)

        # 保存图片
        
        output_filename = os.path.join(dir_name, f"run{run_number}_{label}_plot.png")
        plt.savefig(output_filename, dpi=150)
        plt.close()  # 关闭图形以释放内存
        print(f"图像已保存为: {output_filename}")

if __name__ == "__main__":
    main()