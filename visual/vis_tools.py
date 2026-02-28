from numpy.random.mtrand import f
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

class DrawHist():
    def __init__(self, name):
        self.name = name
        self.bins = 255

    def __call__(self, collect_files, save_path):
        for file in collect_files:
            data = np.load(file)
            file_stem = Path(file).stem
            data = data.flatten()
            self.draw_hist(data, file_stem, save_path=os.path.join(save_path, file_stem + '_hist_add.png'), add_mean=True)
        
    def draw_hist(self, data, title, save_path='./hist.png', add_mean=False):
        plt.figure(figsize=(10, 8))
        n, bins, patches = plt.hist(data, bins=self.bins)
        # 保存图片
        if add_mean:
            n_mean = n.mean()
            n_add = []
            for i in n:
                if i>0:
                    n_add.append(i+n_mean)
                else:
                    n_add.append(0)
            plt.bar(bins[:-1], n_add, width=np.diff(bins), align='edge', label='hist')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

class DrawChn3D():
    def __init__(self, name):
        self.bins = 255
        self.cols = 8
        self.name = name
        self.max = 32

    def __call__(self, collect_files, save_path):
        for file in collect_files:
            data = np.load(file)
            file_stem = Path(file).stem
            n, c = data.shape[0], data.shape[1]
            print(f'file_stem: {file_stem}, shape: {data.shape}')
            if c <= 10 and n == 1:
                self.draw_chn_3d(data, file_stem, save_path=os.path.join(save_path, file_stem + '_chn_3d.png'))
            else:
                self.draw_chn_unfold(data, file_stem, save_path=os.path.join(save_path, file_stem + '_chn_3dunfold.png'))

    def draw_chn_unfold(self, data, title, save_path='./chn_3dunfold.png'):
        """
        可视化PyTorch模型层权重的直方图分布
        
        参数:
        data: torch.Tensor, 形状为 [1, c, h, w]  [c, v]的权重张量
        """
        # 维度只支持4维度，且第一个维度值为1
        if not (data.ndim == 4 or data.ndim == 2):
            print("dims is 4  or  2")
            return
        
        # 将权重从 [1, c, h, w] 转换为 [c, h*w]
        c = data.shape[0]
        n = 1
        if data.ndim == 4:
            n, c, h, w = data.shape
            # 最后两个维度合并
            data = data.reshape(n, c, h*w)
        else:
            data = np.expand_dims(data, axis=0)
            
        # Calculate histograms
        histograms = []
        
        # 最大绘制64个通道，超过64个通道则按间隔抽取
        step = 1
        axes = None
        if c > self.max:
            step = c // self.max
            # Plot histograms
            rows = self.max // self.cols
            fig, axes = plt.subplots(rows, self.cols, figsize=(20, 2*rows))
            axes = axes.flatten()
                    
        else:
            # Plot histograms
            rows = (n*c + self.cols - 1) // self.cols
            fig, axes = plt.subplots(rows, self.cols, figsize=(20, 2*rows))
            axes = axes.flatten()
            
        curr_idx = 0
        draw_idxs = []
        for j in range(n):
            for i in range(c):
                if curr_idx % step == 0:
                    hist, _ = np.histogram(data[j, i], bins=self.bins, density=True)
                    histograms.append(hist)
                    draw_idxs.append(curr_idx)
                curr_idx += 1
                if len(draw_idxs) == self.max:
                    break
            if len(draw_idxs) == self.max:
                break

        for i, hist in enumerate(histograms):
            ax = axes[i]
            idx = draw_idxs[i]
            ax.plot(hist)
            ax.set_title(f'layer {math.floor(idx/c)} chn {idx%c}')
            ax.set_xlabel('bin')
            ax.set_ylabel('Probability')
        
        # Hide unused subplots
        for j in range(c, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def draw_chn_3d(self, data, title, save_path='./chn_3d.png'):
        """
        可视化PyTorch模型层权重的直方图分布
        
        参数:
        data: torch.Tensor, 形状为 [1, c, h, w]  [c, v]的权重张量
        """
        # 维度只支持4维度，且第一个维度值为1
        assert data.ndim == 4 and data.shape[0] == 1 or data.ndim == 2, "dims is 4 and first dim is 1, or dims is 2"
        
        # 将权重从 [1, c, h, w] 转换为 [c, h*w]
        if data.ndim == 4:
            c, h, w = data.shape[1:]
            data = data.squeeze(0).view(c, -1)
        c = data.shape[0]
        
        # Calculate histograms for each channel
        histograms = []
        bin_edges = None
        
        for i in range(c):
            hist, bin_edges = np.histogram(data[i], bins=255, density=True)
            histograms.append(hist)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all channels in 3D
        colors = plt.cm.tab10(np.linspace(0, 1, min(c, 10)))  # 使用不同颜色表示不同通道
        
        for i in range(c):
            # Use a subset of bins for better visualization
            step = 1
            y = bin_edges[:-1][::step]
            x = np.full_like(y, i)
            z = histograms[i][::step]
            
            ax.plot(x, y, z, color=colors[i % len(colors)],  alpha=0.5)
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('bins')
        ax.set_zlabel('Probability Density')
        ax.set_title('3D Visualization of Weight Histograms Across Channels')
                        
        # 调整视角以获得更好的可视化效果
        ax.view_init(elev=30, azim=45)
                
        plt.tight_layout()
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

class VisualTools():
    vis_ability = { 'a_hist' : DrawHist('a_hist'),
                    'w_hist' : DrawHist('w_hist'), 
                    'a_chn_3d' : DrawChn3D('a_chn_3d'),
                    'w_chn_3d' : DrawChn3D('w_chn_3d')}

    # 可视化能力，每个key绑定一个成员函数
    def __init__(self, collect_path, save_path, vis_set={}):
        super(VisualTools, self).__init__()
        self.vis_set = vis_set
        self.save_path = save_path
        self.collect_path = collect_path
        
        all_collect = os.listdir(collect_path)
        self.w_collects = [os.path.join(collect_path, f) for f in all_collect if f.startswith('w_')]
        self.a_collects = [os.path.join(collect_path, f) for f in all_collect if f.startswith('a_')]

    def visualize(self):
        for vis in self.vis_set:
            if vis in self.vis_ability:
                if vis.startswith('w'):
                    self.vis_ability[vis](self.w_collects, self.save_path)
                elif vis.startswith('a'):
                    self.vis_ability[vis](self.a_collects, self.save_path)
                else:
                    print(f"No vis_ability for {vis}")
            else:
                print(f"No vis_ability for {vis}")

