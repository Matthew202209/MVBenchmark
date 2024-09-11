import glob
import os
import pickle

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

input_dir = r"./index/Citadel"
dataset = "nfcorpus"
prune_weight = "0.6"
if __name__ == '__main__':
    expert_dir = r"{}/{}/{}/{}".format(input_dir, dataset, r"expert", prune_weight)
    save_path = r'./Chart'
    input_paths = sorted(glob.glob(os.path.join(expert_dir, "*.pkl")))
    cache = []
    memory = 0
    for input_path in tqdm(input_paths):
        expert_id = int(input_path.split("/")[-1].split(".")[0])
        id_data, _, repr_data = load_file(input_path)
        cache.append((expert_id, id_data, repr_data))
    cache = sorted(cache, key=lambda x: -len(x[2]))
    cpu_end = int(len(cache) * 1)
    cached_experts = {}
    for k, id_data, repr_data in cache[:cpu_end]:
        memory += id_data.nelement() * id_data.element_size() + repr_data.nelement() * repr_data.element_size()
        cached_experts[k] = (id_data.to(torch.int64), repr_data.to(torch.float32))


    chunk_items_num = []
    for k,v in cached_experts.items():
        chunk_items_num.append(v[0].shape[0])

    print(chunk_items_num)
    # 绘制直方图
    # 创建索引列表
    indexes = list(range(len(chunk_items_num)))

    # 绘制条形图
    plt.figure(figsize=(15, 8))  # 设置图形的大小
    plt.bar(indexes, chunk_items_num, color='blue')  # 绘制蓝色的条形图
    plt.title(r'{}'.format(dataset))  # 图表标题
    plt.xlabel('Chunk')  # x轴标签
    plt.ylabel('The Number of Token')  # y轴标签 # 将索引值显示在x轴上，并旋转90度以便阅读 # 只在y轴方向添加网格线
   # 显示图表
    plt.savefig(r'{}/distribution/{}.png'.format(save_path, dataset), dpi=300)  #
    plt.show()

