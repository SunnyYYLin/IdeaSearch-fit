"""
数据加载工具：将 CSV 文件转换为 IdeaSearchFitter 可用的格式
"""
import numpy as np
import pandas as pd


def load_ccp_data():
    """
    加载 CCP (Cubic Close Packing) 数据
    
    输入变量: c (Li浓度), phi (离子势)
    输出变量: sigma (电导率)
    
    Returns:
        dict: 包含 'x' (输入) 和 'y' (输出) 的字典
    """
    df = pd.read_csv('data/ccp.csv')
    
    # 输入: c 和 phi
    x_data = df[['c', 'phi']].values  # shape: (n_samples, 2)
    
    # 输出: sigma
    y_data = df['sigma'].values  # shape: (n_samples,)
    
    return {
        "x": x_data,
        "y": y_data,
    }


def load_hcp_data():
    """
    加载 HCP (Hexagonal Close Packing) 数据
    
    输入变量: c (Li浓度), phi (离子势)
    输出变量: sigma (电导率)
    
    Returns:
        dict: 包含 'x' (输入) 和 'y' (输出) 的字典
    """
    df = pd.read_csv('data/hcp.csv')
    
    # 输入: c 和 phi
    x_data = df[['c', 'phi']].values  # shape: (n_samples, 2)
    
    # 输出: sigma
    y_data = df['sigma'].values  # shape: (n_samples,)
    
    return {
        "x": x_data,
        "y": y_data,
    }


def save_as_npz(data_dict, output_path):
    """
    将数据保存为 .npz 格式（也可以传给 IdeaSearchFitter 的 data_path 参数）
    
    Args:
        data_dict: 包含 'x' 和 'y' 的字典
        output_path: 输出文件路径（如 'data/ccp.npz'）
    """
    np.savez(output_path, x=data_dict['x'], y=data_dict['y'])
    print(f"数据已保存到: {output_path}")


if __name__ == "__main__":
    # 测试加载 CCP 数据
    print("=== CCP 数据 ===")
    ccp_data = load_ccp_data()
    print(f"输入 x 形状: {ccp_data['x'].shape}")
    print(f"输出 y 形状: {ccp_data['y'].shape}")
    print(f"输入变量范围:")
    print(f"  c (Li浓度): [{ccp_data['x'][:, 0].min():.3f}, {ccp_data['x'][:, 0].max():.3f}]")
    print(f"  phi (离子势): [{ccp_data['x'][:, 1].min():.3f}, {ccp_data['x'][:, 1].max():.3f}]")
    print(f"输出变量范围:")
    print(f"  sigma (电导率): [{ccp_data['y'].min():.4f}, {ccp_data['y'].max():.4f}]")
    
    print("\n=== HCP 数据 ===")
    hcp_data = load_hcp_data()
    print(f"输入 x 形状: {hcp_data['x'].shape}")
    print(f"输出 y 形状: {hcp_data['y'].shape}")
    print(f"输入变量范围:")
    print(f"  c (Li浓度): [{hcp_data['x'][:, 0].min():.3f}, {hcp_data['x'][:, 0].max():.3f}]")
    print(f"  phi (离子势): [{hcp_data['x'][:, 1].min():.3f}, {hcp_data['x'][:, 1].max():.3f}]")
    print(f"输出变量范围:")
    print(f"  sigma (电导率): [{hcp_data['y'].min():.4f}, {hcp_data['y'].max():.4f}]")
    
    # 可选：保存为 .npz 格式
    save_as_npz(ccp_data, 'data/ccp.npz')
    save_as_npz(hcp_data, 'data/hcp.npz')
