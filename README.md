# 项目名称 (Project Name)

本项目基于 PyTorch 和 PyTorch Geometric (PyG) 构建。由于图神经网络底层依赖对 PyTorch 版本和 CUDA 版本要求极其严格，请严格按照以下步骤配置虚拟环境。

## 环境要求 (Prerequisites)

- **OS**: Ubuntu22.04
- **Conda**: Anaconda
- **CUDA**: 11.3+

---

## 快速安装指南 (Installation Guide)

### 1. 创建并激活虚拟环境

我们建议将环境安装在项目目录下的 `envs/` 文件夹中：

```bash
conda create -p envs/python39_env python=3.9.7 -y
conda activate ./envs/python39_env

# 创建 python 软链接
ln -s ./envs/python39_env/bin/python python

# 创建 pip 软链接
ln -s ./envs/python39_env/bin/pip pip
```

### 配置国内镜像加速 (可选，针对国内网络)

为了加速 Conda 包的下载，建议配置清华大学开源软件镜像站：

```bash
# 添加主通道镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

# 设置显示包来源 URL
conda config --set show_channel_urls yes
```
### 4. 安装 PyTorch (CUDA 11.3)

本项目依赖 `PyTorch 1.11.0` 及其对应的视觉和音频库：

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```


### 5. 安装 PyTorch Geometric (PyG) 底层依赖

> **注意**：PyG 的底层 C++ 拓展包必须与 PyTorch 和 CUDA 版本严丝合缝。
>
> 如果你更改了前一步的 PyTorch/CUDA 版本，请务必前往 [PyG 官方数据源](https://data.pyg.org/whl/) 寻找对应版本的 whl 文件。

对于 `PyTorch 1.11.0 + cu113` 和 `Python 3.9`，请执行以下脚本离线下载并安装：

```bash
# 创建临时目录
mkdir -p tmp/pyg_whl && cd tmp/pyg_whl

# 下载对应的 whl 包
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/pyg_lib-0.2.0%2Bpt111cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.15-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

# 批量安装
../../pip install *.whl

# 清理临时文件
cd ../../
rm -rf tmp
```

### 6. 安装常规 Python 依赖

最后，使用 `requirements.txt` 一键安装剩余的通用依赖包（包括数据处理、可视化模块和 PyG 的高级 API）：

```bash
./pip install -r requirements.txt
```

运行
./python main.py


### 5. 修复 MKL 相关的 JIT 报错

如果出现了底层 C++ 库冲突导致的 `undefined symbol: iJIT_NotifyEvent` 错误，请强制安装指定版本的 MKL：

```bash
conda install mkl==2024.0 -c conda-forge
```