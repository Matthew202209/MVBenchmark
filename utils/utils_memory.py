import os

import psutil


def memory_usage():
    # 获取当前进程的内存使用情况
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # 返回常驻内存的字节数（RSS）