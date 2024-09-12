import os
import psutil


def memory_usage():
    # 获取当前进程的内存使用情况
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # 返回常驻内存的字节数（RSS）

def get_folder_size(folder_path):
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def colbert_get_folder_size(folder_path, is_colbertv2 = False):
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "metadata" in list(file.split('.')):
                continue
            if "plan" in list(file.split('.')):
                continue
            if file == "ivf.ori.pt":
                if not is_colbertv2:
                    continue
            if file == "ivf.pid.pt":
                if is_colbertv2:
                    continue
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size