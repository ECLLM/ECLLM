import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import gaussian_kde
import json


file_path_list = ['./dataset/input_train/mmlu/boolq_1.json','./dataset/input_train/mmlu/mmlu_1.json','./dataset/input_train/mmlu/drop_1.json','./dataset/input_train/mmlu/gsm8k_1.json']
new_embeds = []
length = [0]
n_com = 1 # tsne dimension


for file_path in file_path_list:
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            instances = json.load(f)
            for i in range(len(instances)):
                if i % 100 == 0:
                    new_data = eval(instances[i]["output_7b_embedding"])
                    new_embeds.append(new_data)
    elif file_path.endswith(".txt"):
        with open(file_path, 'r') as g:
            for i, line in enumerate(g):
                if (i + 1) % 10 == 0:
                    new_data = eval(line.strip())
                    new_embeds.append(new_data)

    length.append(len(new_embeds)-1)
    print(np.array(new_embeds).shape)


tsne = TSNE(n_components=n_com, random_state=42)
embedded_data = tsne.fit_transform(np.array(new_embeds))
print(len(embedded_data),len(embedded_data[0]))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
print(length)
color_list = ['g','r','y','b','k','c','w','m']
legend_list = ['BoolQ','MMLU','DROP','GSM8k']


if n_com == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(length)-1):
        # plt.scatter(embedded_data[length[i]:length[i+1], 0], embedded_data[length[i]:length[i+1], 1], color = color_list[i], label = legend_list[i])
        ax.scatter(embedded_data[length[i]:length[i+1], 0], embedded_data[length[i]:length[i+1], 1], color = color_list[i], label = legend_list[i])

    # font = FontProperties(family='Times New Roman', size=12)
    # ax.legend(prop=font)
    # ax.legend(fontsize=16)
    # ax.set_xlabel('',fontsize=16)
    # ax.set_ylabel('',fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    plt.xlabel('Dimension 1 of tSNE', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('Dimension 2 of tSNE', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
    plt.subplots_adjust(bottom=0.15, left=0.15)


    # ax.set_title('t-SNE Visualization')
    # plt.xlabel('X-axis', fontsize=14)
    # plt.ylabel('Y-axis', fontsize=16)
    plt.savefig('./dataset/input_train/mmlu/distribution_2.png',dpi=300)

if n_com == 1:
    for i in range(len(length)-1):
        # kde = gaussian_kde(embedded_data[length[i]:length[i+1]])
        # x = np.linspace(min)
        # plt.scatter(embedded_data[length[i]:length[i+1], 0], embedded_data[length[i]:length[i+1], 1], color = color_list[i], label = legend_list[i])
        flaten_data = [item[0] for item in embedded_data]

        print(flaten_data[length[i]:length[i+1]])
        sns.kdeplot(flaten_data[length[i]:length[i+1]], fill=True, color=color_list[i], label=legend_list[i])

    # font = FontProperties(family='Times New Roman', size=12)
    # ax.legend(prop=font)
        
    # plt.legend(fontsize=16)
    # plt.ylabel('')
    # plt.tick_params(axis='both', labelsize=16)

    plt.xlabel('Dimension 1 of tSNE', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('Distribution Density', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
    plt.subplots_adjust(bottom=0.15, left=0.15)

    # ax.set_title('t-SNE Visualization')
    # plt.savefig('./dataset/7b_input/mmlu/input_distribution_1.png',dpi=300)
    plt.savefig('./',dpi=300)