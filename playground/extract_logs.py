import os
import re
import shutil
import tarfile
import time
import argparse
import math

def extract_number(filename):
    """从文件名中提取数字部分并返回整数."""
    match = re.search(r'model_(\d+).pt', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # 如果没有匹配到数字，返回-1

def is_file_older_than_n_days(file_path, n):
    """检查文件的最后修改时间是否早于n天前."""
    if n == 0:  # 如果n_days为0，表示不跳过任何文件
        return False
    file_mtime = os.path.getmtime(file_path)
    current_time = time.time()
    time_diff = current_time - file_mtime
    return time_diff > (n * 86400)  # 86400秒 = 1天

def clear_directory(directory):
    """清空目录中的所有内容."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # 删除文件或符号链接
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 删除目录
    print(f"Cleared directory: {directory}")

def main(n_days, max_id, output_dir):
    print("Starting script...")
    logs_dir = 'logs'
    
    # 检查 logs 目录是否存在
    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} is not a directory.")
        return
    
    # 检查 output 目录是否存在以及是否非空
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            print(f"Error: {output_dir} is not a directory.")
            return
        if os.listdir(output_dir):  # 如果目录非空
            print(f"Warning: {output_dir} is not empty.")
            user_input = input("Do you want to clear the output directory? (y/n): ").strip().lower()
            if user_input == 'y':
                clear_directory(output_dir)
            else:
                print("Script aborted by user.")
                return
        else:
            print(f"{output_dir} exists and is empty.")
    else:
        print(f"{output_dir} does not exist. Creating it...")
        os.makedirs(output_dir)
    
    # 初始化摘要字典
    summary = {}
    total_copied = 0
    
    # 遍历 logs 目录下的所有 AAA 目录
    print(f"Processing AAA directories in {logs_dir}...")
    for aaa in os.listdir(logs_dir):
        aaa_path = os.path.join(logs_dir, aaa)
        if not os.path.isdir(aaa_path):
            print(f"Skipping {aaa_path} as it is not a directory.")
            continue  # 跳过非目录文件
        
        print(f"Processing AAA directory: {aaa_path}")
        
        # 创建对应的 output/AAA 目录
        aaa_output_path = os.path.join(output_dir, aaa)
        if not os.path.exists(aaa_output_path):
            os.makedirs(aaa_output_path)
            print(f"Created directory: {aaa_output_path}")
        else:
            print(f"Directory already exists: {aaa_output_path}")
        
        # 初始化 AAA 的摘要
        if aaa not in summary:
            summary[aaa] = {}
        
        # 遍历 AAA 目录下的所有 BBB 目录
        print(f"Processing BBB directories in {aaa_path}...")
        for bbb in os.listdir(aaa_path):
            bbb_path = os.path.join(aaa_path, bbb)
            if not os.path.isdir(bbb_path):
                print(f"Skipping {bbb_path} as it is not a directory.")
                continue  # 跳过非目录文件
            
            print(f"Processing BBB directory: {bbb_path}")
            
            # 创建对应的 output/AAA/BBB 目录
            bbb_output_path = os.path.join(aaa_output_path, bbb)
            if not os.path.exists(bbb_output_path):
                os.makedirs(bbb_output_path)
                print(f"Created directory: {bbb_output_path}")
            else:
                print(f"Directory already exists: {bbb_output_path}")
            
            # 查找所有 model_***.pt 文件
            print(f"Searching for model_*.pt files in {bbb_path}...")
            files = [f for f in os.listdir(bbb_path) if f.startswith('model_') and f.endswith('.pt')]
            
            if not files:
                print(f"No model_*.pt files found in {bbb_path}. Skipping.")
                continue  # 如果没有匹配的文件，跳过
            
            # 提取所有文件的数字并找到最大值
            numbers = [extract_number(f) for f in files]
            numbers = [n for n in numbers if n <= max_id]  # filter the picked number to be not greater than max_id
            if not numbers:
                print(f"No valid model_*.pt files in {bbb_path}")
                continue
            
            max_number = max(numbers)
            
            # 检查最大编号是否有效
            if max_number == -1:
                print(f"Warning: No valid model_*.pt files in {bbb_path}")
                continue
            
            target_file = f"model_{max_number}.pt"
            src_file = os.path.join(bbb_path, target_file)
            
            # 检查文件是否在n天前创建
            if is_file_older_than_n_days(src_file, n_days):
                print(f"Skipping {target_file} as it was created more than {n_days} days ago.")
                continue
            
            if target_file in files:
                dst_file = os.path.join(bbb_output_path, target_file)
                print(f"Copying file from {src_file} to {dst_file}")
                shutil.copy(src_file, dst_file)
                # 更新摘要
                summary[aaa][bbb] = target_file
                total_copied += 1
            else:
                print(f"Warning: {target_file} not found in {bbb_path}")

    # 打印摘要
    print("\nSummary of copied files:")
    for aaa, bbb_dict in summary.items():
        print(f"\nAAA: {aaa}")
        for bbb, file in bbb_dict.items():
            print(f"    BBB: {bbb}, File: {file}")
    print(f"\nTotal files copied: {total_copied}")
    
    # Remove all empty directories in output/
    print("Removing empty directories in output/...")
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                # print(f"Removed empty directory: {dir_path}")
    
    if os.path.exists(output_dir):
        # Create a tar.gz archive of the output/ directory
        print(f"Creating {output_dir}.tar.gz...")
        with tarfile.open(f"{output_dir}.tar.gz", 'w:gz') as tar:
            tar.add(output_dir, arcname=output_dir)
        print(f"{output_dir}.tar.gz has been created.")
    else:
        print(f"{output_dir} does not exist. No archive created.")
    print("Finished script.")

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Copy model files and skip files older than n days.")
    parser.add_argument(
        '--n_days',
        type=int,
        default=0,
        help="Skip files older than n days. Default is 0 (do not skip any files)."
    )
    parser.add_argument(
        '--max_id',
        type=int,
        default=math.inf,
        help="Filter the picked number to be not greater than this value. Default is inf (do not filter)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help="Output directory. Default is 'output'."
    )
    args = parser.parse_args()
    
    # 调用主函数并传入n_days和max_id参数
    main(args.n_days, args.max_id, args.output_dir)