
## Wavelet-based Mamba with Fourier Adjustment for Low-light Image Enhancement(WalMaFa)


Junhao Tan, Songwen Pei, Wei Qin, Bo Fu, Ximing Li and Libo Huang

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2410.20314)

>**Abstract:** Frequency information (e.g., Discrete Wavelet Transform and Fast Fourier Transform) has been widely applied to solve the issue of Low-Light Image Enhancement (LLIE). However, existing frequency-based models primarily operate in the simple wavelet or Fourier space of images, which lacks utilization of valid global and local information in each space. We found that wavelet frequency information is more sensitive to global brightness due to its low-frequency component while Fourier frequency information is more sensitive to local details due to its phase component. In order to achieve superior preliminary brightness enhancement by optimally integrating spatial channel information with low-frequency components in the wavelet transform, we introduce channel-wise Mamba, which compensates for the long-range dependencies of CNNs and has lower complexity compared to Diffusion and Transformer models. So in this work, we propose a novel Wavelet-based Mamba with Fourier Adjustment model called **WalMaFa**, consisting of a Wavelet-based Mamba Block (WMB) and a Fast Fourier Adjustment Block (FFAB). We employ an Encoder-Latent-Decoder structure to accomplish the end-to-end transformation. Specifically, WMB is adopted in the Encoder and Decoder to enhance global brightness while FFAB is adopted in the Latent to fine-tune local texture details and alleviate ambiguity. Extensive experiments demonstrate that our proposed WalMaFa achieves state-of-the-art performance with fewer computational resources and faster speed.

#### News
- **Sep 21, 2024:** Our paper has been accepted by ACCV 2024! :boom: :boom: :boom:
- **Jun 8, 2024:** Pre-trained models are released!
- **Jun 8, 2024:** Codes is released!
- **Jun 8, 2024:** Homepage is released!



![](figures/cover.png)

## Network Architecture
![](figures/network.png)

The overview of the WalMaFa architecture. Our model consists of an Encoder-Latent-Decoder structure that uses wavelet-based WMB to adjust global brightness during the Encoder and Decoder, and Fourier-based FFAB to adjust local details during the Latent.

## Module Design
![](figures/module.png)
## Qualitative results
### Results on LOL datasets

![](figures/LOL_experiment.png)

### Results on non-reference datasets
![](figures/unpaired.png)




## Get Started
### Dependencies and Installation
1. Create Conda Environment 
```bash
conda create -n WalMaFa python=3.10
conda activate WalMaFa

# 进入到libfile文件夹
cd libfile
下载文件：https://pan.baidu.com/s/1Pagb5RrYiC84JAyyEh_sPg?pwd=phfe
pip install torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl 
pip install torchvision-0.18.1+cu118-cp310-cp310-linux_x86_64.whl 
pip install torchaudio-2.3.1+cu118-cp310-cp310-linux_x86_64.whl

pip install triton==2.3.1
pip install transformers==4.43.3

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc

pip install causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

git clone https://github.com/luo3300612/Visualizer.git
cd Visualizer
pip install bytecode
python setup.py install
    
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm einops tensorboard pyyaml pytorch-msssim warmup_scheduler tensorboardX easydict torchmetrics
```

以下不需要执行
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install cudatoolkit==11.7 -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
conda install packaging

cudnn安装：(不需要操作)
    conda search cudnn
    conda install cudnn=8.9.2
    conda list | grep -E "cudatoolkit|cudnn" #查看cudnn版本
    cat /usr/lib/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 #查看cudnn安装版本

设置 CUDA_HOME 为 Conda 环境的 CUDA 路径
export CUDA_HOME=$CONDA_PREFIX
将 CUDA 库路径添加到动态链接库搜索路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

gcc版本切换方法：(GCC版本冲突时使用)
    gcc -v #查看gcc版本
    ls /usr/bin/gcc* #查看已安装gcc文件
    sudo update-alternatives --config gcc #切换gcc

visualizer安装方法： 
    pip uninstall visualizer
    git clone https://github.com/luo3300612/Visualizer.git
    cd Visualizer
    pip install bytecode
    python setup.py install

mamba_ssm安装方法：
    git clone https://github.com/Dao-AILab/causal-conv1d.git 
    cd causal-conv1d 
    git checkout v1.2.0 # current latest version tag 
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
    cd ..
    git clone https://github.com/state-spaces/mamba.git
    cd ./mamba
    git checkout v1.2.0 # current latest version tag
    MAMBA_FORCE_BUILD=TRUE pip install .

```
2. Clone Repo
```
git clone https://github.com/mcpaulgeorge/WalMaFa.git
```


### Pretrained Model
We provide the pre-trained models:
- WalMaFa trained on LOL [[Google drive](https://drive.google.com/drive/folders/1wEVqm5Z9tKCLqN6SAEYwetQMk-ViCCSz?usp=sharing) | [Baidu drive](https://pan.baidu.com/s/1j5KwGHWxMsaPwHP2u5Vj7A?pwd=5zyt)]




### Test
You can directly test the pre-trained model as follows

1. Modify the paths to dataset and pre-trained mode. 
```python
# Tesing parameter 
input_dir # the path of data
result_dir # the save path of results 
weights # the weight path of the pre-trained model
```

2. Test the models for LOL dataset

You need to specify the data path ```input_dir```, ```result_dir```, and model path ```weight_path```. Then run
```bash
python test.py --input_dir your_data_path --result_dir your_save_path --weights weight_path

```

### Train

1. To download datasets training and testing data

2.  To train WalMaFa, run
```bash
python train.py -yml_path your_config_path
```


### Reference Repositories
This implementation is based on / inspired by:
- LLFormer: https://github.com/TaoWangzj/LLFormer
- RetinexFormer: https://github.com/caiyuanhao1998/Retinexformer
- SNR: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
- IAT: https://github.com/cuiziteng/Illumination-Adaptive-Transformer

### Citation
If you find WalMaFa helpful, please cite our paper:
```
@InProceedings{Tan_2024_ACCV,
    author    = {Tan, Junhao and Pei, Songwen and Qin, Wei and Fu, Bo and Li, Ximing and Huang, Libo},
    title     = {Wavelet-based Mamba with Fourier Adjustment for Low-light Image Enhancement},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {3449-3464}
}
```
---



