# 关于 IdeaSearch-fit

## 快速开始

运行以下指令以安装 IdeaSearch-fit：

```bash
pip install IdeaSearch-fit
```

## 项目概述

`IdeaSearch-fit` 是一款**基于 `IdeaSearch` 框架、专注于符号回归的科学公式发现工具**。它提供了 `IdeaSearchFitter` 类，该类被设计为一个高度集成的“助手 (Helper)”对象，通过 `IdeaSearch` 的 `bind_helper` 方法无缝对接。`IdeaSearch-fit` 融合了进化计算与大型语言模型的双重能力，旨在从实验或模拟数据中自动发现潜在的数学表达式。

`IdeaSearch-fit` 的核心任务是解决符号回归问题：给定一组输入数据 `X` 及其对应的输出 `y`，系统会通过进化搜索，寻找一个能够精确描述它们关系，同时自身又足够简洁的数学公式 `f(X)`。因此，它不仅是一个数据拟合工具，其最终目标是成为一个能够揭示数据背后物理规律或数学模式的探索性框架。

## 核心方法论：启发式搜索与数值优化的结合

`IdeaSearch-fit` 的核心在于其混合式工作流程，该流程将大型语言模型（LLM）的宏观结构探索与传统数值方法的微观参数优化相结合：

1.  **第一阶段：基于大型语言模型的拟设 (Ansatz) 搜索**
    此阶段利用 **LLM 的指令遵循能力和模式识别能力，在由数学符号、物理概念和自然语言构成的空间中，进行启发式的候选公式搜索**。LLM 根据用户提供的数据特征、物理单位及可选的理论背景描述，生成一系列结构合理的候选数学表达式（即“拟设”）。这是一个宏观的、探索性的结构生成过程。

2.  **第二阶段：基于数值优化的拟设参数求解**
    在获得一个候选拟设后（例如 `param1 * sin(param2 * x)`），`IdeaSearchFitter` 实例会立即利用**用户指定的传统数值优化算法（如 `L-BFGS-B` 或 `differential-evolution`）**，来精确求解该拟设中的待定参数（`param1`, `param2` 等）的最优值。这是一个确定性的、微观的参数寻优过程，旨在精确评估每个候选结构的拟合能力。

这种“LLM 探索结构，数值优化求解参数”的工作流，整合了前沿人工智能的结构生成能力与经典算法的精确计算能力，构成了 `IdeaSearch-fit` 的基础。

## 核心输入与输出

从用户角度，`IdeaSearch-fit` 的工作流被设计得非常直接：

> **核心输入:**
>
> - **`X`**: 输入数据（自变量，支持多维）。
> - **`y`**: 目标数据（因变量，一维）。
> - **`error`** (可选): `y` 中每个数据点的测量误差。在科学数据拟合中，提供误差是进行严谨性评估（如使用约化卡方检验）的基础。
>
> **核心输出:**
>
> - 一个**帕累托前沿 (Pareto Frontier)**。系统产出的不是单一的最优公式，而是一个解的集合。此集合中的每个公式都在其各自的复杂度水平上达到了最优的拟合精度，从而允许用户在模型的**简洁性**与**精确性**之间做出符合领域需求的权衡。

## 关键特性

- **进化式符号回归**: 依托 `IdeaSearch` 的多岛屿并行进化框架为核心引擎，在广阔的数学表达式空间中进行高效搜索，以发现最优的拟合公式。

- **帕累托最优前沿**: `IdeaSearch-fit` 不会止步于寻找单一的最佳拟合，而是同时优化**精度（Metric）与简洁度（Complexity）这两个相互制约的目标。它最终会生成一份帕累托前沿 (Pareto Frontier)** 报告，提供了一系列在不同复杂度与精度组合下的最优解，供用户深入分析和权衡。

- **严格的量纲分析**: 内置量纲验证模块 (`unit_validator`)，能够在公式的进化过程中进行严格的物理单位检查。此特性对于物理、工程等科学领域至关重要，它确保了最终生成的公式不仅在数值上成立，在物理意义上也同样正确。

- **双模态公式生成**:

  - **精确生成 (Precise Generation)**: 直接生成和进化严格遵循特定计算语法的数学表达式。
  - **模糊生成 (Fuzzy Generation)**: 首先，利用大型语言模型（LLM）像科学家一样，提出关于数据潜在规律的**自然语言理论**；然后，由另一个 LLM 代理将该理论**翻译**为严格的数学表达式。这种创新的两阶段方法能够融入更丰富的先验知识并激发创造性思路。

- **自动化智能润色**: 用户仅需提供基础的变量名和单位，系统即可调用 LLM 自动推断并补全对输入、输出及各变量物理含义的详细描述。这些信息将用于构建更具语义的提示词，引导 `IdeaSearch` 进行更深层次的智能探索。

- **高度集成与自动化**: 作为 `IdeaSearch` 的一个高度集成的模块，用户只需初始化 `IdeaSearchFitter` 类，它就能自动配置好 `IdeaSearch` 所需的评估函数 (`evaluate_func`)、提示词 (`prologue_section`, `epilogue_section`)、变异/交叉算子 (`mutation_func`, `crossover_func`) 等全部核心组件。

## 核心 API

`IdeaSearch-fit` 包的主要交互接口是 `IdeaSearchFitter` 类，它由构造函数和几个结果获取方法组成。

- `IdeaSearchFitter(__init__)`: ⭐️ **重要**

  - **功能**: 初始化一个 `IdeaSearchFitter` 实例。这是所有配置的统一入口，涵盖了数据加载、问题定义、表达式构建、搜索策略设定等所有环节。
  - **重要性**: 使用 `IdeaSearch-fit` 的第一步，决定了整个符号回归任务的框架。

- `get_best_fit()`:

  - **功能**: 从所有已评估的公式中，返回经最优参数拟合后的数值表达式字符串。该表达式在所有公式中达到了最低的度量值（如均方误差）。
  - **重要性**: 获取单一维度的最佳拟合结果。

- `get_pareto_frontier()`:

  - **功能**: 返回一个字典，包含了构成当前帕累托前沿的所有公式及其详细信息（复杂度、度量值、拟合参数等）。
  - **重要性**: 获取一系列在精度和复杂度之间取得最优权衡的拟合结果。

## 配置参数详解

`IdeaSearchFitter` 实例的所有配置都在其 `__init__` 构造函数中完成。参数已按逻辑功能划分，以便于理解和配置。

### 1\. 任务输入与输出

- **`data`**: `Optional[Dict[str, ndarray]]`
  - 以字典形式直接传入内存中的数据。必需包含键 `"x"` (输入, 2D array) 和 `"y"` (输出, 1D array)，可选包含 `"error"` (y 的误差, 1D array)。
- **`data_path`**: `Optional[str]`
  - 从本地 `.npz` 文件加载数据，与 `data` 参数二选一。文件内应包含与 `data` 参数相同的键。
- **`result_path`**: `str`
  - 一个**已存在的文件夹路径**，用于存储拟合过程的产出，如帕累托前沿报告 (`pareto_report.txt`) 和数据 (`pareto_data.json`)。

### 2\. 问题定义与量纲

- **`variable_names`**: `Optional[List[str]]`
  - 输入变量的名称列表 (如 `["mass", "velocity"]`)。若不提供，则默认为 `["x1", "x2", ...]`.
- **`output_name`**: `Optional[str]`
  - 输出变量的名称 (如 `"energy"`)。
- **`perform_unit_validation`**: `bool` (默认: `False`)
  - **[核心开关]** 是否开启严格的物理量纲验证。
- **`variable_units`**: `Optional[List[str]]`
  - 与 `variable_names` 对应的单位列表 (如 `["kg", "m s^-1"]`)。当 `perform_unit_validation` 为 `True` 时必需。
- **`output_unit`**: `Optional[str]`
  - 输出变量的单位 (如 `"J"` 或 `"kg m^2 s^-2"`)。当 `perform_unit_validation` 为 `True` 时必需。
- **`auto_polish`**: `bool` (默认: `True`)
  - 是否自动调用 LLM 来推断并生成变量的物理含义描述，以构建语义更丰富的提示词。
- **`auto_polisher`**: `Optional[str]`
  - 用于执行 `auto_polish` 任务的 LLM 模型名称。若为 `None`，将自动选择可用模型。
- **`input_description`**, **`variable_descriptions`**, **`output_description`**: `Optional[str]`
  - 手动指定对整个系统、输入变量、输出变量的文字描述。如果提供了这些信息，`auto_polish` 步骤将被跳过。

### 3\. 表达式构建模块

- **`functions`**: `List[str]` (默认: 包含 `sin`, `cos`, `log`, `exp` 等常用函数)
  - 允许在表达式中使用的数学函数列表。所有函数必须受 `numexpr` 库支持。
- **`constant_whitelist`**: `List[str]` (默认: `[]`)
  - 允许在表达式中使用的常数名称列表（字符串形式）。
- **`constant_map`**: `Dict[str, float]` (默认: `{"pi": np.pi}`)
  - 将常数名称映射到其浮点数值的字典。

### 4\. 搜索与进化策略

- **LLM 生成策略**
  - **`generate_fuzzy`**: `bool` (默认: `True`)
    - **[核心开关]** 是否启用“模糊生成”模式。若为 `False`，LLM 将直接生成严格的数学表达式。
  - **`fuzzy_translator`**: `Optional[str]`
    - 在“模糊生成”模式下，用于将自然语言理论翻译成数学表达式的 LLM 模型名称。
- **数值优化策略**
  - **`optimization_method`**: `Literal["L-BFGS-B", "differential-evolution"]` (默认: `"L-BFGS-B"`)
    - 指定用于求解拟设参数的数值优化算法。`L-BFGS-B` 是一种高效的局部优化算法，适用于较为平滑的目标函数。`differential-evolution` 是一种全局优化算法，对于存在多个局部最优解的复杂问题可能更具鲁棒性。
  - **`optimization_trial_num`**: `int` (默认: `100`)
    - 每次拟合时，数值优化器的尝试次数。对于 `differential-evolution`，这对应于种群大小。对于非凸或复杂问题，增加此值可以提高找到全局最优解的概率，但会增加计算时间。
- **进化算子（开发中）**
  - **`enable_mutation`**: `bool` (默认: `False`)
    - 是否启用内置的表达式**变异**功能。
  - **`enable_crossover`**: `bool` (默认: `False`)
    - 是否启用内置的表达式**交叉**功能。

### 5\. 评分映射

- **`baseline_metric_value`**: `Optional[float]`
  - 定义一个“基准”度量值（如 MSE），该值将被映射为 `IdeaSearch` 的分数 `20.0`。若为 `None`，将使用朴素线性拟合的结果作为基准。
- **`good_metric_value`**: `Optional[float]`
  - 定义一个“良好”度量值，该值将被映射为 `IdeaSearch` 的分数 `80.0`。
- **`metric_mapping`**: `Literal["linear", "logarithm"]` (默认: `"linear"`)
  - 度量值到分数的映射方式。对数 (`logarithm`) 映射更适用于度量值数量级变化剧烈的情况。

### 6\. 杂项

- **`seed`**: `Optional[int]`
  - 随机数种子，用于确保结果的可复现性。

## 工作流程

以下是一个完整的工作流程示例（[https://github.com/IdeaSearch/IdeaSearch-fit-test](https://github.com/IdeaSearch/IdeaSearch-fit-test)），展示了如何使用 `IdeaSearch-fit` 和 `IdeaSearch` 解决一个符号回归问题。

```python
# pip install IdeaSearch-fit
import numpy as np
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def build_data():
    """
    构建用于拟合的模拟数据。
    真实公式: x = 1.2*A + A*exp(-0.7*gamma*t) - A*exp(-0.5*gamma*t)*cos(3*omega*t)
    """
    n_samples = 1000
    rng = np.random.default_rng(seed=42)

    # 定义自变量的取值范围
    A_range = (1.0, 10.0)
    gamma_range = (0.1, 1.0)
    omega_range = (1.0, 5.0)
    t_range = (0.0, 10.0)

    # 在范围内随机采样
    A_samples = rng.uniform(A_range[0], A_range[1], n_samples)
    gamma_samples = rng.uniform(gamma_range[0], gamma_range[1], n_samples)
    omega_samples = rng.uniform(omega_range[0], omega_range[1], n_samples)
    t_samples = rng.uniform(t_range[0], t_range[1], n_samples)

    # 将自变量组合成 Fitter 需要的格式 (n_samples, n_features)
    x_data = np.stack([A_samples, gamma_samples, omega_samples, t_samples], axis=1)

    # 根据真实公式计算 y 值，并加入噪声
    y_true = (1.2 * A_samples +
              A_samples * np.exp(-0.7 * gamma_samples * t_samples) -
              A_samples * np.exp(-0.5 * gamma_samples * t_samples) * np.cos(3 * omega_samples * t_samples))
    error_data = 0.01 + 0.02 * np.abs(y_true)
    y_data = y_true + rng.normal(0, error_data)

    # 注意：本示例未将 error_data 传给 Fitter，因此将使用 MSE 作为度量
    return {
        "x": x_data,
        "y": y_data,
    }


def main():
    # 1. 准备数据
    data = build_data()

    # 2. 初始化 IdeaSearchFitter
    fitter = IdeaSearchFitter(
        result_path = "fit_results", # 确保此文件夹已创建
        data = data,
        variable_names = ["A", "gamma", "omega", "t"],
        variable_units = ["m", "s^-1", "s^-1", "s"],
        output_name = "x",
        output_unit = "m",
        constant_whitelist = ["1", "2", "pi"],
        constant_map = {"1": 1, "2": 2, "pi": np.pi},
        auto_polish = True,
        generate_fuzzy = True,
        perform_unit_validation = True,
        optimization_method = "L-BFGS-B",
        optimization_trial_num = 5,
    )

    # 3. 初始化 IdeaSearcher
    ideasearcher = IdeaSearcher()

    # 4. 基础配置
    ideasearcher.set_program_name("IdeaSearch Fitter Test")
    ideasearcher.set_database_path("database") # 确保此文件夹已创建
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models(["gemini-2.5-pro"]) # 使用您期望的模型

    ideasearcher.set_record_prompt_in_diary(True) # 可选配置：在日志中记录提示词

    # 5. 绑定 Fitter！ (最关键的一步)
    ideasearcher.bind_helper(fitter)

    # 6. 定义并执行进化循环
    island_num = 3
    cycle_num = 3
    unit_interaction_num = 20

    for _ in range(island_num):
        ideasearcher.add_island()

    for cycle in range(cycle_num):
        print(f"---[ 周期 {cycle + 1}/{cycle_num} ]---")
        if cycle != 0: ideasearcher.repopulate_islands()
        ideasearcher.run(unit_interaction_num)
        print("完成。")

    # 7. 获取并展示结果
    print("\n---[ 搜索完成 ]---")
    print(f"最佳拟合公式: {fitter.get_best_fit()}")

    print("\n帕累托前沿上的最优解集:")
    pareto_frontier = fitter.get_pareto_frontier()
    if not pareto_frontier:
        print("  - 未在帕累托前沿上找到解。")
    else:
        for complexity, info in sorted(pareto_frontier.items()):
            metric_key = "reduced chi squared" if "reduced chi squared" in info else "mean square error"
            metric_value = info.get(metric_key, float('nan'))
            print(f"  - 复杂度: {complexity}, 误差: {metric_value:.4g}, 公式: {info.get('ansatz', 'N/A')}")


if __name__ == "__main__":
    # 在运行前，请确保已创建 `./fit_results` 和 `./database` 文件夹
    # 并已正确配置 `api_keys.json` 文件
    main()
```
