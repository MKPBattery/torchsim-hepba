import os
import urllib.request
import zipfile
from pathlib import Path

def download_mace_model():
    """
    下载MACE模型并设置环境
    """
    # 创建模型目录
    model_dir = Path('models/mace')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载模型文件
    model_url = "https://github.com/ACEsuit/mace-models/raw/main/models/mace_mp.pt"
    model_path = model_dir / "mace_mp.pt"
    
    if not model_path.exists():
        print("正在下载MACE模型...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"模型已下载到: {model_path}")
    else:
        print("MACE模型已存在")
    
    return str(model_path.absolute())

if __name__ == "__main__":
    model_path = download_mace_model()
    print(f"MACE模型路径: {model_path}") 