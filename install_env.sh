#!/bin/bash
# 遇到错误立即退出脚本
set -e

echo "========================================================="
echo "开始配置多模态推荐系统运行环境..."
echo "========================================================="

# 1. 确保 conda 命令在脚本的子 shell 中可用
eval "$(conda shell.bash hook)"

# 2. 创建并激活虚拟环境
ENV_DIR="envs/python39_env"
echo -e "\n>>> 正在创建 Conda 虚拟环境: $ENV_DIR (Python 3.9.7)"
conda create -p $ENV_DIR python=3.9.7 -y
conda activate ./$ENV_DIR

# 3. 配置国内镜像加速 (清华源)
echo -e "\n>>> 配置 Conda 清华镜像源..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free || true
conda config --set show_channel_urls yes

# 4. 修复 MKL 相关的 JIT 报错
echo -e "\n>>> 安装 MKL 2024.0 以修复 undefined symbol 报错..."
conda install mkl==2024.0 -c conda-forge -y

# 5. 安装 PyTorch (CUDA 11.3)
echo -e "\n>>> 安装 PyTorch 1.11.0 (CUDA 11.3)..."
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# 6. 离线下载并安装 PyTorch Geometric (PyG) 的 C++ 拓展包
echo -e "\n>>> 下载并安装 PyG 离线拓展包 (torch-1.11.0+cu113)..."
mkdir -p tmp/pyg_whl
cd tmp/pyg_whl

# 加上 -q --show-progress 让下载界面更清爽
wget -q --show-progress https://data.pyg.org/whl/torch-1.11.0%2Bcu113/pyg_lib-0.2.0%2Bpt111cu113-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.15-cp39-cp39-linux_x86_64.whl
wget -q --show-progress https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl

echo -e "\n>>> 开始安装 whl 文件..."
pip install *.whl

cd ../../
rm -rf tmp

# 7. 安装剩余的 Python 依赖包 (附带 pip 国内镜像)
echo -e "\n>>> 安装通用依赖包 (NumPy, Pandas, PyG 等)..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    tqdm==4.67.3 \
    numpy==1.26.4 \
    pandas==2.3.2 \
    scipy==1.13.1 \
    pyyaml==6.0.2 \
    lmdb==1.7.3 \
    torch-geometric==2.5.1 \
    matplotlib==3.9.4 \
    seaborn==0.13.2

echo -e "\n========================================================="
echo "🎉 环境配置已全部成功完成！"
echo "👉 请在您的终端中运行以下命令手动进入环境："
echo "conda activate ./envs/python39_env"
echo "========================================================="
