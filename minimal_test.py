# coding: utf-8
"""
最小化测试，避免依赖问题
"""

import sys
import os
sys.path.append('/home/ljy/Documents/nwu/science/code/TT2/src')

def minimal_test():
    """最小化测试"""
    print("开始最小化测试...")
    
    try:
        # 直接测试模型文件
        import torch
        model_path = 'saved/TT6/TT6-baby-Oct-22-2025-11-28-13.pth'
        
        if os.path.exists(model_path):
            print(f"✅ 模型文件存在: {model_path}")
            
            # 尝试加载模型文件
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✅ 模型文件加载成功")
            
            if 'model_state_dict' in checkpoint:
                print("✅ 包含model_state_dict")
            if 'epoch' in checkpoint:
                print(f"训练轮数: {checkpoint['epoch']}")
            if 'valid_score' in checkpoint:
                print(f"验证分数: {checkpoint['valid_score']}")
                
            return True
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    minimal_test()


