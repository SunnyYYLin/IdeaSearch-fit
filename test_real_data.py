"""
使用真实数据 (CCP/HCP) 进行公式拟合测试
"""
import numpy as np
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter
from load_data import load_ccp_data, load_hcp_data

SYSTEM_DESCRIPTION = """Analyze the relationship between Li+ concentration (c), ionic potential (phi), and ionic conductivity (σ) in a Li+ {crystal_structure} crystalline material.

**Task**: Propose a formula for σ = f(c, φ).

**Requirements**:
1. Neat: do not include unnecessary complexity to prevent overfitting.
2. Physically meaningful: start from first principles.
3. Empirically valid: ensure the formula is consistent with available data.
"""

def test_ccp_data():
    """测试 CCP (Cubic Close Packing) 数据拟合"""
    print("=" * 60)
    print("开始测试 CCP 数据拟合")
    print("=" * 60)
    
    # 1. 加载数据
    data = load_ccp_data()
    
    # 2. 初始化 IdeaSearchFitter
    fitter = IdeaSearchFitter(
        result_path="results",
        data=data,
        variable_names=["c", "phi"],
        variable_units=["mol L^-1", "V"],  # c为浓度(mol/L), phi为离子势(伏特)
        output_name="sigma",
        output_unit="S m^-1",  # 电导率单位: 西门子/米
        constant_whitelist=["1", "2", "pi"],
        constant_map={"1": 1, "2": 2, "pi": np.pi},
        input_description=SYSTEM_DESCRIPTION.format(crystal_structure="CCP"),
        auto_polish=False,
        generate_fuzzy=True,
        fuzzy_translator="gpt-4o",  # 用于将理论翻译成数学公式的模型
        perform_unit_validation=False,  # 如果都是无量纲可以关闭
        optimization_method="L-BFGS-B",
        optimization_trial_num=10,
    )
    
    # 3. 初始化 IdeaSearcher
    ideasearcher = IdeaSearcher()
    ideasearcher.set_program_name("CCP Data Fitting Test")
    ideasearcher.set_database_path("database")
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models(["gpt-5-mini", "gpt-4o"])  # 用于生成理论草稿的模型
    ideasearcher.set_record_prompt_in_diary(True)
    
    # 4. 绑定 Fitter
    ideasearcher.bind_helper(fitter)
    
    # 5. 执行进化搜索
    island_num = 2
    cycle_num = 2
    unit_interaction_num = 10
    
    for _ in range(island_num):
        ideasearcher.add_island()
    
    for cycle in range(cycle_num):
        print(f"\n---[ Cycle {cycle + 1}/{cycle_num} ]---")
        if cycle != 0:
            ideasearcher.repopulate_islands()
        ideasearcher.run(unit_interaction_num)
        print("Done.")
    
    # 6. 显示结果
    print("\n" + "=" * 60)
    print("CCP 数据拟合结果")
    print("=" * 60)
    try:
        print(f"最佳拟合公式: {fitter.get_best_fit()}")
        
        print("\n帕累托前沿解集:")
        pareto_frontier = fitter.get_pareto_frontier()
        if not pareto_frontier:
            print("  - 未找到有效解")
        else:
            for complexity, info in sorted(pareto_frontier.items()):
                metric_key = "reduced chi squared" if "reduced chi squared" in info else "mean square error"
                metric_value = info.get(metric_key, float('nan'))
                print(f"  - 复杂度: {complexity}, 误差: {metric_value:.4g}")
                print(f"    公式: {info.get('ansatz', 'N/A')}")
    except Exception as e:
        print(f"获取结果时出错: {e}")


def test_hcp_data():
    """测试 HCP (Hexagonal Close Packing) 数据拟合"""
    print("\n" + "=" * 60)
    print("开始测试 HCP 数据拟合")
    print("=" * 60)
    
    # 1. 加载数据
    data = load_hcp_data()
    
    # 2. 初始化 IdeaSearchFitter
    fitter = IdeaSearchFitter(
        result_path="results/hcp_task_requirements_more_cycles",
        data=data,
        variable_names=["c", "phi"],
        variable_units=["mol L^-1", "V"],  # c为浓度(mol/L), phi为离子势(伏特)
        output_name="sigma",
        output_unit="S m^-1",  # 电导率单位: 西门子/米
        constant_whitelist=["1", "2", "pi"],
        constant_map={"e": np.e, "1": 1, "2": 2, "pi": np.pi, "kB": 1.380649e-23, "T": 300},
        input_description=SYSTEM_DESCRIPTION.format(crystal_structure="HCP"),
        auto_polish=False,
        generate_fuzzy=True,
        fuzzy_translator="gpt-4o",  # 用于将理论翻译成数学公式的模型
        perform_unit_validation=False,
        optimization_method="L-BFGS-B",
        optimization_trial_num=10,
    )
    
    # 3. 初始化 IdeaSearcher
    ideasearcher = IdeaSearcher()
    ideasearcher.set_program_name("HCP Data Fitting Test")
    ideasearcher.set_database_path("database")
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models(["gpt-5-mini", "gpt-4o", "gpt-4.1"])  # 用于生成理论草稿的模型
    ideasearcher.set_record_prompt_in_diary(True)
    
    # 4. 绑定 Fitter
    ideasearcher.bind_helper(fitter)
    
    # 5. 执行进化搜索
    island_num = 5
    cycle_num = 30
    unit_interaction_num = 3
    
    for _ in range(island_num):
        ideasearcher.add_island()
    
    for cycle in range(cycle_num):
        print(f"\n---[ Cycle {cycle + 1}/{cycle_num} ]---")
        if cycle != 0:
            ideasearcher.repopulate_islands()
        ideasearcher.run(unit_interaction_num)
        print("Done.")
    
    # 6. 显示结果
    print("\n" + "=" * 60)
    print("HCP 数据拟合结果")
    print("=" * 60)
    try:
        print(f"最佳拟合公式: {fitter.get_best_fit()}")
        
        print("\n帕累托前沿解集:")
        pareto_frontier = fitter.get_pareto_frontier()
        if not pareto_frontier:
            print("  - 未找到有效解")
        else:
            for complexity, info in sorted(pareto_frontier.items()):
                metric_key = "reduced chi squared" if "reduced chi squared" in info else "mean square error"
                metric_value = info.get(metric_key, float('nan'))
                print(f"  - 复杂度: {complexity}, 误差: {metric_value:.4g}")
                print(f"    公式: {info.get('ansatz', 'N/A')}")
    except Exception as e:
        print(f"获取结果时出错: {e}")


if __name__ == "__main__":
    # 选择要测试的数据集
    import sys
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = "ccp"  # 默认测试 CCP
    
    if dataset == "ccp":
        test_ccp_data()
    elif dataset == "hcp":
        test_hcp_data()
    elif dataset == "both":
        test_ccp_data()
        test_hcp_data()
    else:
        print(f"未知数据集: {dataset}")
        print("用法: python test_real_data.py [ccp|hcp|both]")
