import numpy as np
import numexpr
import re
from typing import (
    List, Tuple,
)


__all__ = [
    "DIMENSION_NAMES",
    "validate_unit",
]


DIMENSION_NAMES = [
    "m", 
    "s", 
    "kg", 
    "T", 
    "V",
]


def formalize(unit: str) -> List[float]:
    # 基本单位列表和索引
    base_units = ['m', 's', 'kg', 'K', 'V']
    unit_vector = [0.0] * 5
    
    # 常见派生单位映射到基本单位表达式
    common_units = {
        "J": "kg m^2 s^-2",
        "Pa": "kg m^-1 s^-2",
        "N": "kg m s^-2",
        "W": "kg m^2 s^-3",
        "Hz": "s^-1",
        "C": "kg m^2 s^-2 V^-1",
        "A": "kg m^2 s^-3 V^-1",
        # 可以添加更多
    }
    
    if unit == "Dimensionless":
        return unit_vector

    # 首先，替换派生单位
    # 将输入字符串按空格分割，然后每个单词如果是派生单位则替换，然后重新组合
    words = unit.split()
    expanded_words = []
    for word in words:
        if word in common_units:
            expanded_words.extend(common_units[word].split())
        else:
            expanded_words.append(word)
    unit_str = " ".join(expanded_words)
    
    # 现在unit_str应该是基本单位表达式
    tokens = unit_str.split()
    for token in tokens:
        # 检查是否有^
        if '^' in token:
            parts = token.split('^')
            unit_name = parts[0]
            exp = float(parts[1])
        else:
            unit_name = token
            exp = 1.0
        
        if unit_name in base_units:
            idx = base_units.index(unit_name)
            unit_vector[idx] += exp
        else:
            raise ValueError(f"未知单位: {unit_name}")
    
    return unit_vector

def validate_unit(
    expression: str,
    expression_unit: str,
    variable_names: List[str],
    variable_units: List[str],
)-> Tuple[bool, str]:
    
    """
    通过数值方法检查表达式的量纲一致性。

    该方法基于量纲齐次性原理。它为每个基本量纲应用一系列缩放因子，
    并检查表达式的输出是否遵循幂律。如果遵循幂律，则其幂次与期望量纲进行比较。
    如果不遵循幂律，则很可能表达式中存在量纲不兼容的加法或减法。

    Args:
        expression (str): 数学表达式字符串 (例如, "param1 * x / y")。
                          此方法假设表达式中的参数 (param1, param2等) 是无量纲常数。
        expression_unit (List[float]): 表达式结果的期望量纲向量。
        variable_names (List[str]): 变量名列表。
        variable_units (List[List[float]]): 与变量名对应的量纲向量列表。

    Returns:
        Tuple[bool, str]: 一个元组，包含:
        - 布尔值，指示量纲是否一致。
        - 字符串，详细说明检查结果。
    """
    
    zero_threshold = 1e-9                   # 防止除以零
    r_threshold = 1e-9                      # 回归系数的计算误差
    num_test_cases = 100                    # 原始测试用例数
    scaling_factors = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]      # 缩放因子
        )
    random_seed = 42                        # 随机数种子

    expression_unit_formal = formalize(expression_unit)
    variable_units_formal = [formalize(u) for u in variable_units]

    input_scale = dict(zip(variable_names, variable_units_formal))

    # 1. 从表达式中识别变量和参数
    variables = sorted(list(input_scale.keys()))
    # 使用正则表达式查找 "param" 后跟数字的参数
    params = sorted(list(set(re.findall(r"param\d+", expression))))

    # 2. 为所有变量和参数生成多个随机的基准值
    np.random.seed(random_seed) # 设置随机种子
    test_cases = []
    for _ in range(num_test_cases):
        local_dict = {}
        for name in variables + params:
            # 使用更大的范围（-1到6）以避免特定值的问题
            local_dict[name] = np.random.uniform(-1, 6)
        test_cases.append(local_dict)

    # 3. 筛选有效的测试用例（表达式计算有效且不为零）
    valid_cases = []
    for local_dict in test_cases:
        try:
            result = numexpr.evaluate(expression, local_dict=local_dict)
            if not (np.isnan(result) or np.isinf(result) or np.isclose(result, 0)):
                valid_cases.append((local_dict, result))
        except:
            continue

    if not valid_cases:
        return False, "所有测试用例均无效（计算错误、为零、无穷或NaN）。无法进行缩放检查。"

    # 使用第一个有效用例作为基准
    base_dict, base_result = valid_cases[0]

    # 4. 逐个检查每个量纲的缩放情况
    inconsistent_dims = []
    num_dims = len(DIMENSION_NAMES)

    for i in range(num_dims):
        # 对当前维度应用一系列缩放因子
        actual_results = []
        for s in scaling_factors:
            scaled_local_dict = base_dict.copy()
            # 根据第 i 个维度的幂次，对每个变量进行缩放
            for var in variables:
                power = input_scale[var][i]
                scaled_local_dict[var] *= (s ** power)
            
            # 使用缩放后的输入值计算表达式
            try:
                scaled_result = numexpr.evaluate(expression, local_dict=scaled_local_dict)
                actual_results.append(scaled_result)
            except Exception as e:
                return False, f"对维度 '{DIMENSION_NAMES[i]}' 进行缩放计算时出错: {e}"

        actual_scalings = np.array(actual_results) / base_result

        # 检查缩放后结果是否正常
        if np.any(actual_scalings < 0) or np.any(np.isnan(actual_scalings)) or np.any(np.isinf(actual_scalings)):
            inconsistent_dims.append(
                f"维度 '{DIMENSION_NAMES[i]}' 不一致。表达式在该维度上不遵循幂律缩放, "
                f"且缩放时表达式符号改变或出现异常行为，"
                f"可能存在量纲不兼容的运算。"
            )
            continue  # 继续检查下一个维度

        # 检查是否遵循幂律关系
        # 使用 log-log 图的线性关系来判断
        log_s = np.log(scaling_factors)
        log_a = np.log(actual_scalings)

        if len(log_s) > 1:
            # 检查线性度
            if np.std(log_a) <= zero_threshold:
                r_squared = 1.0
            else:
                r_squared = np.corrcoef(log_s, log_a)[0, 1]**2
            if r_squared < 1-r_threshold:
                inconsistent_dims.append(
                    f"维度 '{DIMENSION_NAMES[i]}' 不一致。表达式在该维度上不遵循幂律缩放, "
                    f"但是缩放时表达式符号不变，"
                    f"可能存在量纲不兼容的运算。"
                )
                continue  # 继续检查下一个维度

            # 如果是幂律，计算实际幂次
            p = np.polyfit(log_s, log_a, 1)
            actual_power = p[0]
        else:
            # 如果只有一个缩放点，则退化为原方法
            actual_power = log_a[0] / log_s[0]

        # 将计算出的实际幂次与期望幂次进行比较
        theoretical_power = expression_unit_formal[i]
        if not np.isclose(actual_power, theoretical_power):
            inconsistent_dims.append(
                f"维度 '{DIMENSION_NAMES[i]}' 不一致。期望幂次: {theoretical_power}, "
                f"但表达式表现出的实际幂次为: {actual_power:.2f}"
            )

    # 5. 报告最终结果
    if not inconsistent_dims:
        return True, "表达式量纲一致。"
    else:
        error_message = "\n".join(inconsistent_dims)
        return False, f"表达式量纲不一致:\n{error_message}"

