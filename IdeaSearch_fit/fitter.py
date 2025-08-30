import os
import numexpr
import warnings
import numpy as np
from numpy import ndarray
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Callable
from typing import Optional
from threading import Lock
from numpy.random import default_rng
from pywheels.math_funcs import reduced_chi_squared
from pywheels.math_funcs import mean_squared_error
from pywheels.blueprints.ansatz import Ansatz
from pywheels.blueprints.ansatz import ansatz_docstring
from .unit_validator import validate_unit
from .i18n import translate


__all__ = [
    "IdeaSearchFitter",
]


_numexpr_supported_functions: List[str] = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", 
    "cosh", "tanh", "log", "log10", "exp", "sqrt", "abs", 
]


class IdeaSearchFitter:
    
    # ------------------------- IdeaSearchFitter初始化 -------------------------
    
    def __init__(
        self,
        data: Optional[Dict[str, ndarray]] = None,
        data_path: Optional[str] = None,
        existing_fit: str = "0.0",
        functions: List[str] = _numexpr_supported_functions.copy(),
        perform_unit_validation: bool = False,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
        output_unit: Optional[str] = None,
        baseline_metric_value: Optional[float] = None, # metric value corresponding to score 20.0
        good_metric_value: Optional[float] = None, # metric value corresponding to score 80.0
        metric_mapping: Literal["linear", "logarithm"] = "linear",
        auto_rescale: bool = False,
        adjust_degrees_of_freedom: bool = False,
        enable_mutation: bool = False,
        enable_crossover: bool = False,
        seed: Optional[int] = None,
    )-> None:
        
        self._preflight_check(
            data = data,
            data_path = data_path,
            perform_unit_validation = perform_unit_validation,
            variable_names = variable_names,
            variable_units = variable_units,
            output_unit = output_unit,
        )
        
        self._random_generator = default_rng(seed)
        
        self._existing_fit: str = existing_fit
        self._perform_unit_validation: bool = perform_unit_validation
        self._auto_rescale: bool = auto_rescale
        self._adjust_degrees_of_freedom: bool = adjust_degrees_of_freedom
        
        self._initialize_data(data, data_path)
        self._process_data()
        self._analyze_data()
        
        self._set_variables(variable_names, variable_units)
        self._set_functions(functions); self._set_prompts()
        self._output_unit: Optional[str] = output_unit
        
        self._set_naive_linear_idea(); self._set_initial_ideas()
        
        self._build_numexpr_dict()
        self._build_metric_mapper(
            baseline_metric_value = baseline_metric_value,
            good_metric_value = good_metric_value,
            metric_mapping = metric_mapping,
        )
        
        # hijack action `mutation_func` and `crossover_func` when disabled
        if not enable_mutation: self.mutation_func = None # type: ignore
        if not enable_crossover: self.crossover_func = None # type: ignore
        
        self._best_fit: Optional[str] = None
        self._best_metric_value: float = float("inf")
        self._best_fit_lock: Lock = Lock()
    
    # ----------------------------- 外部动作 -----------------------------

    def evaluate_func(
        self,
        idea: str
    )-> Tuple[float, Optional[str]]:

        try:
            
            ansatz = Ansatz(
                expression = idea,
                variables = self._variables,
                functions = self._functions,
            )
            
            if self._perform_unit_validation:
                
                assert self._output_unit is not None
                assert self._variable_units is not None
            
                unit_correctness, unit_validation_info = validate_unit(
                    expression = idea,
                    expression_unit = self._output_unit,
                    variable_names = self._variables,
                    variable_units = self._variable_units,
                )
                
                if not unit_correctness:
                    score = 0.0
                    info = f"拟设量纲错误！具体信息：{unit_validation_info}"
                    return score, info

            best_params, best_metric_value = self._get_idea_optimal_result(
                idea = idea,
            )
            
            best_params_msg = ""
            
            for index, best_param in enumerate(best_params):
                best_params_msg += f"  param{index+1}: {best_param:.8g}"
                
            self._update_best_fit(
                expression = idea,
                best_params = best_params,
                best_metric_value = best_metric_value,
            )
            
            metric_type = "reduced chi squared" \
                if (self._error is not None) else "mean square error"
            
            ansatz_param_num = ansatz.get_param_num()
            
            y_pred_remainder_rescaled = numexpr.evaluate(
                ex = ansatz.reduce_to_numeric_ansatz(best_params), 
                local_dict = self._numexpr_local_dict,
            )
                
            if self._error is not None:
                
                true_metric_value = reduced_chi_squared(
                    predicted_data = y_pred_remainder_rescaled * self._y_rescale_factor + self._existing_fit_value,
                    ground_truth_data = self._y,
                    errors = self._error,
                    adjust_degrees_of_freedom = self._adjust_degrees_of_freedom,
                    param_num = ansatz_param_num,
                )
                
            else:
                
                true_metric_value = mean_squared_error(
                    predicted_data = y_pred_remainder_rescaled * self._y_rescale_factor + self._existing_fit_value,
                    ground_truth_data = self._y,
                    adjust_degrees_of_freedom = self._adjust_degrees_of_freedom,
                    param_num = ansatz_param_num,   
                )
                
            residual_report = self._analyze_residuals(
                y_true = self._y_remainder_rescaled,
                y_pred = y_pred_remainder_rescaled,
            )
                
            score = self._metric_mapper(best_metric_value)
            info = (
                f"得分：{score:.2f}\n"
                f"{metric_type}：{true_metric_value:.8g}\n"
                f"最优参数：{best_params_msg}"
                f"最优参数下拟合模型的残差分析：\n{residual_report}"
            )
            
            return score, info
        
        except Exception as error:
            return -1.0, f"拟合出错：{error}"
        
        
    def mutation_func(
        self,
        idea: str,
    )-> str:

        ansatz = Ansatz(
            expression = idea,
            variables = self._variables,
            functions = self._functions,
            seed = self._random_generator.integers(0, 1 << 30),
        )
        
        ansatz.mutate()
        
        return ansatz.to_expression()
        
        
    def crossover_func(
        self,
        parent1: str,
        parent2: str,
    )-> str:
        
        coin = self._random_generator.uniform(0.0, 1.0)
        
        try:
            
            ansatz1 = Ansatz(
                expression = parent1,
                variables = self._variables,
                functions = self._functions,
            )
            
            ansatz2 = Ansatz(
                expression = parent2,
                variables = self._variables,
                functions = self._functions,
            )
            
            if coin < 0.3:
                
                sum_ansatz = ansatz1 + ansatz2
                return sum_ansatz.to_expression()
            
            elif coin < 0.6:
                
                product_ansatz = ansatz1 * ansatz2
                return product_ansatz.to_expression()
            
            elif coin < 0.8:
                
                quotient_ansatz = ansatz1 / ansatz2
                return quotient_ansatz.to_expression()
            
            else:
                
                quotient_ansatz = ansatz2 / ansatz1
                return quotient_ansatz.to_expression()

        except Exception as _:
            return parent1
        
        
    def get_best_fit(
        self,
    )-> str:
        
        best_fit = self._best_fit
        
        if best_fit is None:
            
            raise RuntimeError(
                translate(
                    "【IdeaSearchFitter】无法返回最佳拟合函数，请先尝试运行 IdeaSearch！"
                )
            )
            
        if self._existing_fit != "0.0": best_fit = self._existing_fit + best_fit
                        
        return best_fit
    
    # ----------------------------- 内部动作 -----------------------------
    
    def _preflight_check(
        self,
        data: Optional[Dict[str, ndarray]],
        data_path: Optional[str],
        perform_unit_validation: bool = False,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
        output_unit: Optional[str] = None,
    )-> None:
        
        if (data is None and data_path is None) or \
            (data is not None and data_path is not None):
                
            raise ValueError(translate(
                "【IdeaSearchFitter】初始化时出错：应在 data 与 data_path 间选择一个参数传入！"
            ))
            
        if perform_unit_validation and \
            (variable_names is None or variable_units is None or output_unit is None):
                
            raise ValueError(translate(
                "【IdeaSearchFitter】初始化时出错：单位检查开启时必须传入 variable_names 、 variable_units 和 output_unit！"
            ))
    
    
    def _initialize_data(
        self,
        data: Optional[Dict[str, ndarray]],
        data_path: Optional[str],
    )-> None:
        
        """
        set self._x, self._y, self._error
        """
         
        self._x: ndarray
        self._y: ndarray
        self._error: Optional[ndarray] = None
            
        if data is not None:
            
            if "x" not in data:
                
                raise ValueError(translate(
                    "【IdeaSearchFitter】初始化时出错：data 应包含键 `x` ！"
                ))
                
            if "y" not in data:
                
                raise ValueError(translate(
                    "【IdeaSearchFitter】初始化时出错：data 应包含键 `y` ！"
                ))
                
            self._x = data["x"]; self._y = data["y"]
            
            if "error" in data: self._error = data["error"]
                
        else:
            
            assert data_path is not None

            if not os.path.exists(data_path):
                
                raise FileNotFoundError(
                    translate(
                        "【IdeaSearchFitter】初始化时出错：文件 %s 不存在！"
                    ) % (data_path)
                )
            
            if not data_path.lower().endswith('.npz'):
                
                raise ValueError(translate(
                    "【IdeaSearchFitter】初始化时出错：只支持 .npz 格式文件！"
                ))

            try:
                
                with np.load(data_path) as npz_data:
                    
                    if "x" not in npz_data:
                        
                        raise ValueError(translate(
                            "【IdeaSearchFitter】初始化时出错：npz 文件应包含键 `x` ！"
                        ))
                        
                    if "y" not in npz_data:
                        
                        raise ValueError(translate(
                            "【IdeaSearchFitter】初始化时出错：npz 文件应包含键 `y` ！"
                        ))
                    
                    self._x = npz_data["x"]; self._y = npz_data["y"]
                    if "error" in npz_data: self._error = npz_data["error"]
                    
            except Exception as error:
                
                raise RuntimeError(translate(
                        "【IdeaSearchFitter】初始化时出错：加载 %s 失败 - %s"
                    ) % (data_path, str(error))
                )
                
        if self._x.ndim != 2 or self._y.ndim != 1 \
            or (self._error is not None and self._error.ndim != 1):
                
            raise RuntimeError(translate(
                "【IdeaSearchFitter】初始化时出错：数据形状不合要求，输入数据应为 2 维，输出数据与误差（若有）应为 1 维！"
            ))
            
        if self._y.shape[0] != self._x.shape[0] \
            or (self._error is not None and self._error.shape[0] != self._x.shape[0]):
                
            raise RuntimeError(translate(
                "【IdeaSearchFitter】初始化时出错：数据形状不合要求，输入数据、输出数据与误差（若有）应形状相同！"
            ))
            
            
    def _process_data(
        self,
    )-> None:
        
        """
        set self._input_dim, self._x_rescaled, self._existing_fit_value, 
        self._y_remainder, self._y_remainder_rescaled
        """
    
        self._input_dim: int = self._x.shape[1]
        
        self._x_rescale_factor = self._x.mean(0) if self._auto_rescale else 1
        self._x_rescaled: ndarray = self._x / self._x_rescale_factor
        
        self._existing_fit_value: ndarray = numexpr.evaluate(
            ex = self._existing_fit,
            local_dict = {
                f"x{i + 1}": self._x[:, i]
                for i in range(self._input_dim)
            }
        )
        
        self._y_remainder: ndarray = self._y - self._existing_fit_value
        self._y_rescale_factor = self._y_remainder.mean(0) if self._auto_rescale else 1
        self._y_remainder_rescaled: ndarray = self._y_remainder / self._y_rescale_factor
        
    
    # [Warning] unexamined implementation via vibe coding
    def _analyze_data(
        self
    )-> None:

        n_samples, _ = self._x_rescaled.shape
        y_data = self._y_remainder_rescaled
        
        report_lines = []
        
        x_min = self._x_rescaled.min(axis=0)
        x_max = self._x_rescaled.max(axis=0)
        
        for i, var_name in enumerate(self._variables):
            report_lines.append(f"{var_name}: 范围 [{x_min[i]:.4f}, {x_max[i]:.4f}]")
        
        for i, var_name in enumerate(self._variables):
            x_dim = self._x_rescaled[:, i]
            min_idx = np.argmin(x_dim)
            max_idx = np.argmax(x_dim)
            report_lines.append(f"{var_name}: 极值位置 {min_idx}, {max_idx}")
        
        report_lines.append("")
        
        y_min = y_data.min()
        y_max = y_data.max()
        report_lines.append(f"输出范围: [{y_min:.4f}, {y_max:.4f}]")
        
        y_min_idx = np.argmin(y_data)
        y_max_idx = np.argmax(y_data)
        report_lines.append(f"输出极值位置: {y_min_idx}, {y_max_idx}")
        
        if n_samples > 1:
            y_diff = np.diff(y_data)
            avg_slope = np.mean(np.abs(y_diff))
            max_slope = np.max(np.abs(y_diff))
            min_slope = np.min(np.abs(y_diff))
            
            report_lines.append(f"斜率: 平均{avg_slope:.4f} 最大{max_slope:.4f} 最小{min_slope:.4f}")
            
            if n_samples > 2:
                platform_threshold = avg_slope * 0.1
                platforms = []
                current_platform = []
                
                for i, slope in enumerate(np.abs(y_diff)):
                    if slope < platform_threshold:
                        if not current_platform:
                            current_platform = [i]
                        else:
                            current_platform.append(i)
                    else:
                        if current_platform and len(current_platform) > 1:
                            platforms.append((current_platform[0], current_platform[-1] + 1))
                        current_platform = []
                
                if current_platform and len(current_platform) > 1:
                    platforms.append((current_platform[0], current_platform[-1] + 1))
                
                if platforms:
                    platform_info = []
                    for start, end in platforms:
                        platform_length = end - start + 1
                        platform_avg = np.mean(y_data[start:end+1])
                        platform_info.append(f"{start}-{end}(长{platform_length}均{platform_avg:.3f})")
                    
                    report_lines.append(f"平台: {', '.join(platform_info)}")
        
        report_lines.append("")
        report_lines.append(f"样本数: {n_samples}")
        report_lines.append(f"输出均值: {np.mean(y_data):.4f}")
        report_lines.append(f"输出标准差: {np.std(y_data):.4f}")
        
        self._data_info = "\n".join(report_lines)
        

    # [Warning] unexamined implementation via vibe coding
    def _analyze_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:

        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        residual_std = np.std(residuals)
        
        report_lines = []
        
        mse = np.mean(residuals ** 2)
        mae = np.mean(abs_residuals)
        max_error = np.max(abs_residuals)
        
        report_lines.append(f"拟合误差: MSE={mse:.3e}, MAE={mae:.3e}, 最大误差={max_error:.3e}")
        
        skewness = np.mean(residuals ** 3) / (residual_std ** 3)
        residual_mean = np.mean(residuals)
        
        if abs(skewness) > 0.5:
            bias_direction = "系统性高估" if residual_mean < 0 else "系统性低估"
            report_lines.append(f"⚠️ 检测到{ bias_direction }，建议添加常数项修正")
        else:
            report_lines.append("✓ 残差分布基本对称，无显著系统偏差")
        
        kurtosis = np.mean(residuals ** 4) / (residual_std ** 4) - 3
        if kurtosis > 1:
            report_lines.append("⚠️ 残差呈现厚尾分布，建议考虑指数或对数变换")
        elif kurtosis < -1:
            report_lines.append("⚠️ 残差呈现尖峰分布，建议检查过拟合")
        
        large_error_mask = abs_residuals > 2 * residual_std
        large_error_ratio = np.mean(large_error_mask)
        
        if large_error_ratio > 0.1:
            outlier_positions = np.where(large_error_mask)[0]
            report_lines.append(f"⚠️ 发现{len(outlier_positions)}个异常点，建议增强局部拟合能力")
        
        positive_ratio = np.mean(residuals > 0)
        if abs(positive_ratio - 0.5) > 0.2:
            imbalance_type = "正残差居多" if positive_ratio > 0.5 else "负残差居多"
            report_lines.append(f"⚠️ 残差正负不平衡({imbalance_type})，建议调整函数对称性")
        
        q25, q50, q75 = np.percentile(abs_residuals, [25, 50, 75])
        if q75 / q25 > 3:
            report_lines.append("⚠️ 误差分布离散，建议采用分段函数或加权拟合")
        
        report_lines.append("")
        report_lines.append("改进建议优先级：")
        if abs(skewness) > 0.8:
            report_lines.append("1. 添加常数项或线性项修正系统偏差")
        elif large_error_ratio > 0.15:
            report_lines.append("1. 处理异常点，考虑局部拟合或鲁棒函数")
        elif kurtosis > 1.5:
            report_lines.append("1. 尝试指数/对数变换改善分布")
        else:
            report_lines.append("1. 当前拟合较好，可尝试复杂度更高的函数")
        
        return "\n".join(report_lines)
        
        
    def _set_variables(
        self,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
    )-> None:
        
        """
        set self._variables
        """
        
        self._variables: List[str] = [
            f"x{i + 1}" for i in range(self._input_dim)
        ] if variable_names is None else variable_names
        
        self._variable_units: Optional[List[str]] = variable_units
    
    
    def _set_functions(
        self,
        functions: List[str],
    )-> None:
        
        """
        set self._functions
        """
        
        supported_functions: List[str] = []
        
        for function in functions:
            
            if function in _numexpr_supported_functions:
                supported_functions.append(function)
            
            else:
                warnings.warn(
                    translate(
                        "IdeaSearch-fit 依赖 Python 库 numexpr，而函数 %s 不受 numexpr 支持，已舍去！"
                    ) % (function)
                )
                
        if not supported_functions:
            
            raise RuntimeError(
                translate(
                    "【IdeaSearchFitter】初始化时出错：没有可用的函数！"
                )
            )
        
        self._functions: List[str] = supported_functions   
    
    
    def _set_prompts(
        self,
    )-> None:
        
        """
        configure system_prompt, prologue_section, epilogue_section for IdeaSearcher
        """
        
        prologue_section_variable_string = ", ".join(
            [f'"{variable}"' for variable in self._variables]
        )
        
        prologue_section_function_string = ", ".join(
            [f'"{function}"' for function in self._functions]
        )
        
        variables_info: str
        
        if self._perform_unit_validation:
            
            assert self._variable_units is not None
            assert self._output_unit is not None
            
            variables_info = (
                f"另外， {', '.join(self._variables)} 是物理量，"
                f"单位分别为 {', '.join(self._variable_units)} ，"
                "而所有的 param 都是无量纲（单位为 1 ）的；"
                f"你要搜寻一个单位为 {self._output_unit} 的拟设，请务必确保量纲的正确性。"
            )
        
        else:
        
            variables_info = (
                f"另外，由于 {', '.join(self._variables)} 实际上是物理量，"
                f"我们鼓励你在将 {', '.join(self._variables)} 设为函数自变量前设置一个参数与之相乘。\n"
            )
        
        self.system_prompt: str = (
            "你是一个严格遵循指令、擅长根据已有拟设结构生成具有创新性且符合格式的新拟设表达式的实验科学家。"
            "你会观察已有拟设的表现，从中总结规律，生成既合法又可能更优的 expression 表达式。"
        )
        
        self.prologue_section: str = (
            f"首先，请你阅读以下拟设格式规则：{ansatz_docstring}\n"
            "在本任务中，你只需生成 expression 部分，variables 和 functions 已经固定，分别为：\n"
            f"variables = [{prologue_section_variable_string}]\n"
            f"functions = [{prologue_section_function_string}]\n"
            "请注意这些是你唯一可以使用的变量和函数。\n"
            f"其次，请查看待拟合函数的一份简短报告：\n{self._data_info}\n"
            "然后，为了帮助你理解风格和注意事项，我们先提供一些已有的 expression 示例，请仔细观察它们的结构与潜在问题，总结经验教训后再开始生成：\n"
        )
        
        self.epilogue_section: str = (
            "分析以上示例时，如果你发现某个参数的最优值始终非常接近零，说明它在该任务中作用不大，"
            "你可以考虑在新的拟设中去除这一参数，以提高结构紧凑性与有效性。\n"
            f"{variables_info}"
            "最后，如果遇到非独立参数（例如 param1 * (param2 * x + param3)），"
            "请将其整理为 param1 * x + param2 的形式，让各个参数尽量独立。\n"
            "现在，请你开始生成 expression 表达式。"
            "请记住，直接输出合法的表达式字符串，不要包含解释、注释或额外内容，方便我们接入自动任务流。"
        )
        
        
    def _set_naive_linear_idea(
        self,
    )-> None:
        
        self._naive_linear_idea: str = " + ".join([
            f"param{i + 1} * x{i + 1}"
            for i in range(self._input_dim)
        ] + [f"param{self._input_dim + 1}"])
        
        
    def _set_initial_ideas(
        self,
    )-> None:
        
        """
        configure initial_ideas for IdeaSearcher
        """
        
        self.initial_ideas: List[str] = [self._naive_linear_idea]
    
     
    def _build_numexpr_dict(
        self,
    )-> None:
        
        self._numexpr_local_dict: Dict[str, ndarray] = {
            f"x{i + 1}": self._x_rescaled[:, i]
            for i in range(self._input_dim)
        }
            
            
    def _get_idea_optimal_result(
        self,
        idea: str,
    )-> Tuple[List[float], float]:
        
        ansatz = Ansatz(
            expression = idea,
            variables = self._variables,
            functions = self._functions,
        )
        
        ansatz_param_num = ansatz.get_param_num()

        def numeric_ansatz_user(
            numeric_ansatz: str
        )-> float:
            
            y_pred_remainder_rescaled = numexpr.evaluate(
                ex = numeric_ansatz, 
                local_dict = self._numexpr_local_dict,
            )
            
            if self._error is not None:
                
                metric_value = reduced_chi_squared(
                    predicted_data = y_pred_remainder_rescaled,
                    ground_truth_data = self._y_remainder_rescaled,
                    errors = self._error,
                    adjust_degrees_of_freedom = self._adjust_degrees_of_freedom,
                    param_num = ansatz_param_num,
                )
                
            else:
                
                metric_value = mean_squared_error(
                    predicted_data = y_pred_remainder_rescaled,
                    ground_truth_data = self._y_remainder_rescaled,
                    adjust_degrees_of_freedom = self._adjust_degrees_of_freedom,
                    param_num = ansatz_param_num,   
                )
            
            return metric_value
        
        natural_param_range = (-100.0, 100.0)
                
        best_params, best_metric_value = ansatz.apply_to(
            numeric_ansatz_user = numeric_ansatz_user,
            param_ranges = [natural_param_range] * ansatz_param_num,
            trial_num = 1,
            method = "differential-evolution",
        )
        
        return best_params, best_metric_value
    
    
    def _build_metric_mapper(
        self,
        baseline_metric_value: Optional[float],
        good_metric_value: Optional[float],
        metric_mapping: Literal["linear", "logarithm"],
    )-> None:
        
        if baseline_metric_value is None:
            
            _, baseline = self._get_idea_optimal_result(
                idea = self._naive_linear_idea
            )
            
        else:
            baseline = baseline_metric_value
            
        if good_metric_value is None:
            good = baseline / 10000
            
        else:
            good = good_metric_value
            
        baseline_score = 20.0
        good_score = 80.0
        
        self._metric_mapper: Callable[[float], float] = \
            lambda metric_value: \
            min(100.0,
                max(
                    (good_score - baseline_score) * (np.log(baseline / metric_value)) / (np.log(baseline / good)) + baseline_score \
                    if metric_mapping == "logarithm" else\
                    (good_score - baseline_score) * ((baseline - metric_value) / (baseline - good)) + baseline_score,
                    0.0
                )
            )


    def _update_best_fit(
        self,
        expression: str,
        best_params: List[float],
        best_metric_value: float,
    )-> None:
        
        with self._best_fit_lock:
            
            if best_metric_value >= self._best_metric_value: return
            
            self._best_metric_value = best_metric_value
            
            ansatz = Ansatz(
                expression = expression,
                variables = self._variables,
                functions = self._functions, 
            )
            
            self._best_fit = ansatz.reduce_to_numeric_ansatz(best_params)