#--coding:utf-8--
import torch
from collections import OrderedDict

def load_and_modify_vgg_weights(model_type, weight_path1,weight_path2,weight_path3, new_weight_path):
    """
    读取并修改VGG13或VGG16的权重文件中的键名。

    参数:
        model_type (str): 模型类型，可以是'vgg13'或'vgg16'。
        weight_path (str): 权重文件的路径。
        new_weight_path (str): 保存修改后权重文件的路径。
    """
    # 加载权重文件
    weights1 = torch.load(weight_path1)
    weights2 = torch.load( weight_path2 )
    weights3=torch.load(weight_path3)
    # 创建一个新的OrderedDict来存储修改后的权重
    new_weights = OrderedDict()

    exit()
    # 遍历原始权重字典，修改键名
    for key, value in weights2.items():
        # if key!='classifier.0.weight' and key!='classifier.0.bias' and key!='classifier.3.weight' and key!='classifier.3.bias'and key!='classifier.6.weight' and key!='classifier.6.bias':
        if model_type == 'vgg13':
            new_key=key.replace('features.', '')

        elif model_type == 'vgg16':
            new_key = key.replace('features.', 'vgg.features.')
        new_weights[new_key] = value
    del new_weights['classifier.0.weight']
    del new_weights['classifier.0.bias']
    del new_weights['classifier.3.weight']
    del new_weights['classifier.3.bias']
    del new_weights['classifier.6.weight']
    del new_weights['classifier.6.bias']

    new_weights['25.weight']=weights1[ '31.weight' ]
    new_weights['25.bias']  =weights1[ '31.bias' ]
    new_weights['27.weight']=weights1[ '33.weight' ]
    new_weights['27.bias']  =weights1[ '33.bias' ]

    
    # 保存修改后的权重文件
    torch.save(new_weights, new_weight_path)

# 示例使用
# load_and_modify_vgg_weights('vgg13',weight_path1 ='./vgg16_reducedfc.pth',weight_path2='./vgg16-397923af.pth', weight_path3 = './vgg13_modified.pth',new_weight_path = './vgg16.pth')
load_and_modify_vgg_weights('vgg13',weight_path1 ='./dsfd_40000.pth',weight_path2='./DarkFaceZSDA.pth', weight_path3 = './dsfd_checkpoint.pth',new_weight_path = './vgg13_modified.pth')
# load_and_modify_vgg_weights('vgg13',weight_path1 ='./DarkFaceFS.pth',weight_path2='./vgg13_modified.pth', weight_path3 = './dsfd_70000.pth',new_weight_path = './vgg16.pth')
