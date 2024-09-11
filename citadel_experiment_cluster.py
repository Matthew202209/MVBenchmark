import glob
import os
import pickle
import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

input_dir = r"./index/Citadel"
dataset = "antique"
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


    chunk_sample_embedding = []
    j = 0
    random_numbers = random.sample(range(1, 51), 3)
    tensor_list = [list(cached_experts.values())[i]for i in random_numbers]

    selected_tensors = [random.sample(list(tensor[1]), 100) for tensor in tensor_list]

    chunk_sample_embedding = []
    for selected_tensor in selected_tensors:
        selected_embedding = [item.detach().cpu().numpy() for item in selected_tensor]
        chunk_sample_embedding.append(np.array(selected_embedding))


    colors = ['blue', 'green', 'red']

    # 准备绘图
    plt.figure(figsize=(10, 8))

    # t-SNE处理和绘图
    for sub_list, color in zip(chunk_sample_embedding, colors):
        # 将子列表中的向量转换为二维
        tsne = TSNE(n_components=2, perplexity=min(5, len(sub_list) - 1), random_state=0)
        two_d_vectors = tsne.fit_transform(sub_list)

        plt.scatter(two_d_vectors[:, 0], two_d_vectors[:, 1], color=color, label=f'Chunk {colors.index(color) + 1}')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title(r't-SNE Visualization of random samples for {}'.format(dataset))


    # 显示图表

    plt.savefig(r'{}/cluster/{}-all.png'.format(save_path, dataset), dpi=300)  #
    plt.show()
    for sub_list, color in zip(chunk_sample_embedding, colors):
        plt.figure(figsize=(10, 8))  # 为每个子列表创建一个新的图形

        # t-SNE处理
        tsne = TSNE(n_components=2, perplexity=min(5, len(sub_list) - 1), random_state=0)
        two_d_vectors = tsne.fit_transform(sub_list)

        # 绘制散点图
        plt.scatter(two_d_vectors[:, 0], two_d_vectors[:, 1], color=color, label=f'Chunk {color}')

        # 添加图例
        plt.legend()

        # 添加标题和轴标签
        plt.title(r't-SNE Visualization of {} random samples for {}'.format(color, dataset))

        # 显示图表

        plt.savefig(r'{}/cluster/{}-{}.png'.format(save_path, dataset, color), dpi=300)  #
        plt.show()

    # print(chunk_items_num)
    # # 绘制直方图
    # # 创建索引列表
    # indexes = list(range(len(chunk_items_num)))
    #
    # # 绘制条形图
    # plt.figure(figsize=(15, 8))  # 设置图形的大小
    # plt.bar(indexes, chunk_items_num, color='blue')  # 绘制蓝色的条形图
    # plt.title(r'{}'.format(dataset))  # 图表标题
    # plt.xlabel('Chunk')  # x轴标签
    # plt.ylabel('The Number of Token')  # y轴标签 # 将索引值显示在x轴上，并旋转90度以便阅读 # 只在y轴方向添加网格线
    # plt.show()  # 显示图表
    # plt.savefig(r'{}/distribution/{}.png'.format(save_path, dataset), dpi=300)  #

