"""
测试用户指定公式的拟合效果
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from load_data import load_hcp_data, load_ccp_data


def fit_formula(data, formula_func, param_names, bounds, method="L-BFGS-B"):
    """
    对指定公式进行参数拟合
    
    Args:
        data: 数据字典 {"x": ndarray, "y": ndarray}
        formula_func: 公式函数，接受 (c, phi, params) 返回预测值
        param_names: 参数名称列表
        bounds: 参数边界列表
        method: 优化方法
    
    Returns:
        best_params: 最优参数
        mse: 均方误差
        predictions: 预测值
    """
    c = data['x'][:, 0]
    phi = data['x'][:, 1]
    sigma_true = data['y']
    
    # 定义目标函数（均方误差）
    def objective(params):
        try:
            sigma_pred = formula_func(c, phi, params)
            mse = np.mean((sigma_true - sigma_pred)**2)
            return mse
        except:
            return 1e10  # 返回一个很大的值表示计算失败
    
    # 优化参数
    if method == "L-BFGS-B":
        # 多次随机初始化
        best_result = None
        best_mse = float('inf')

        for trial in range(1000):  # 尝试1000次不同的初始值
            # 随机初始化
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
                if result.fun < best_mse:
                    best_mse = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is None:
            print("优化失败！")
            return None, float('inf'), None
        
        best_params = best_result.x
        mse = best_result.fun
        
    else:  # differential_evolution
        result = differential_evolution(objective, bounds, maxiter=1000)
        best_params = result.x
        mse = result.fun
    
    # 计算预测值
    predictions = formula_func(c, phi, best_params)
    
    return best_params, mse, predictions


def test_user_formula(dataset="hcp"):
    """测试用户指定的公式: sigma = param1*c*exp(param2*phi+param3) + param4"""
    
    print("=" * 70)
    print(f"测试公式: sigma = param1*c*exp(param2*phi+param3) + param4")
    print(f"数据集: {dataset.upper()}")
    print("=" * 70)
    
    # 加载数据
    if dataset.lower() == "hcp":
        data = load_hcp_data()
    else:
        data = load_ccp_data()
    
    c = data['x'][:, 0]
    phi = data['x'][:, 1]
    sigma_true = data['y']
    
    # 定义公式
    def formula(c, phi, params):
        param1, param2, param3, param4 = params
        return param1 * c * np.exp(param2 * phi + param3) + param4
    
    # 参数名称
    param_names = ["param1", "param2", "param3", "param4"]
    
    # 参数边界（根据经验设置）
    bounds = [
        (-10, 10),   # param1
        (-10, 10),   # param2
        (-10, 10),   # param3
        (-10, 10),     # param4
    ]
    
    print("\n优化方法: L-BFGS-B (多次随机初始化)")
    print("-" * 70)
    
    # 拟合
    best_params, mse, predictions = fit_formula(
        data, formula, param_names, bounds, method="L-BFGS-B"
    )
    
    if best_params is None:
        return
    
    # 输出结果
    print("\n拟合结果:")
    print("-" * 70)
    for name, value in zip(param_names, best_params):
        print(f"  {name}: {value:.8f}")
    
    print(f"\n均方误差 (MSE): {mse:.6f}")
    
    # 计算其他指标
    mae = np.mean(np.abs(sigma_true - predictions))
    max_error = np.max(np.abs(sigma_true - predictions))
    r2 = 1 - np.sum((sigma_true - predictions)**2) / np.sum((sigma_true - np.mean(sigma_true))**2)
    
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"最大误差: {max_error:.6f}")
    print(f"R² 分数: {r2:.6f}")
    
    # 生成简单的误差统计
    print("\n" + "=" * 70)
    print("预测误差分布:")
    print("-" * 70)
    residuals = sigma_true - predictions
    print(f"残差均值: {np.mean(residuals):.6f}")
    print(f"残差标准差: {np.std(residuals):.6f}")
    print(f"残差范围: [{np.min(residuals):.6f}, {np.max(residuals):.6f}]")
    
    # 显示部分预测结果
    print("\n" + "=" * 70)
    print("部分预测结果 (前10个样本):")
    print("-" * 70)
    print(f"{'c':<8} {'phi':<8} {'True σ':<12} {'Pred σ':<12} {'Error':<10}")
    print("-" * 70)
    if predictions is not None:
        for i in range(min(10, len(c))):
            error = sigma_true[i] - predictions[i]
            print(f"{c[i]:<8.3f} {phi[i]:<8.4f} {sigma_true[i]:<12.4f} {predictions[i]:<12.4f} {error:<10.4f}")
    
    print("\n" + "=" * 70)
    print("拟合完成！")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = "hcp"
    
    test_user_formula(dataset)
