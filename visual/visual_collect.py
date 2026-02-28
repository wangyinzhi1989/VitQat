import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

def str_to_tensor(s):
    # 将字符串转换为字符列表
    char_list = [ord(c) for c in s]
    # 转换为Tensor
    tensor = torch.tensor(char_list, dtype=torch.int32)
    return tensor

def tensor_to_str(tensor):
    # 将Tensor转换回字符串
    char_list = [chr(int(x)) for x in tensor]
    return ''.join(char_list)

def act_draw_hook(module, input, output):
    if module._buffers['run_num'] is None:
        module._buffers['run_num'] = 1
        module._buffers['act_avg'] = output.detach().cpu().numpy()
    else:
        module._buffers['act_avg'] = (module._buffers['act_avg'] * module._buffers['run_num'] + output.detach().cpu().numpy()) / (module._buffers['run_num'] + 1)
        module._buffers['run_num'] += 1
    #print(f"{tensor_to_str(module._buffers['vis_name'])} act_avg:{module._buffers['act_avg'].shape}  run_num:{module._buffers['run_num']} mean:{module._buffers['act_avg'].mean()}")

class VisualCollect():
    g_act_layers = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.SiLU, nn.Hardswish)
    g_weight_layers = (nn.Conv2d, nn.Linear)

    def __init__(self, model, save_path):
        super(VisualCollect, self).__init__()
        self.model = model
        self.save_path = save_path
        # 遍历网络graph，在需要显示的层注册hook
        self.vis_handle = []
        self.act_layers = dict()
        self.weight_layers = dict()

        for name, module in self.model.named_modules():
            print(f"name:{name}  module:{module}  type:{type(module)}")
            if self.is_act(module):
                self.vis_handle.append(module.register_forward_hook(act_draw_hook))
                module.register_buffer('act_avg', None)
                module.register_buffer('run_num', None)
                module.register_buffer('vis_name', str_to_tensor(name))
                self.act_layers[name] = module

            if self.is_weight(module):
                self.weight_layers[name] = module

        print(f"ah:{self.act_layers} \nwl:{self.weight_layers}")

    def is_act(self,module):
        for act_layer in self.g_act_layers:
            if isinstance(module, act_layer):
                return True
        return False

    def is_weight(self,module):
        for weight_layer in self.g_weight_layers:
            if isinstance(module, weight_layer):
                return True
        return False
        
    def save_weight(self):
        for name, module in tqdm(self.weight_layers.items(), desc="vis_weight", total=len(self.weight_layers)):
            # 绘制权重的分布,并显示每个分布的数目
            w = module.weight.detach().cpu().numpy()
            # 展平w
            #w = w.flatten()
            save_path=os.path.join(self.save_path, f"w_{name}.npy")
            np.save(save_path, w)
            
    def save_activate(self):
        for name, module in tqdm(self.act_layers.items(), desc="vis_activate", total=len(self.act_layers)):
            a = module._buffers['act_avg']
            # 对第0维度进行平均
            a = a.mean(axis=0) #.flatten()
            save_path=os.path.join(self.save_path, f"a_{name}.npy")
            np.save(save_path, a)

    def vis_clear(self):
        for handle in self.vis_handle:
            handle.remove()
