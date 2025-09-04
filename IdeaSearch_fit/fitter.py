import os
import re
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
from pywheels.llm_tools import get_answer
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
    "cosh", "tanh", "log", "log10", "exp", "square", "sqrt", "abs", 
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
        input_description: Optional[str] = None,
        variable_descreption: Optional[Dict[str, str]] = None,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
        output_descreption: Optional[str] = None,
        output_name: Optional[str] = None,
        output_unit: Optional[str] = None,
        generate_fuzzy: bool = False,
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
        self._generate_fuzzy: bool = generate_fuzzy
        
        self._output_unit: Optional[str] = output_unit
        self._output_name: Optional[str] = output_name
        self._variable_descreption: Optional[Dict[str,str]] = variable_descreption
        self._output_descreption: Optional[str] = output_descreption
        self._input_description: Optional[str] = input_description

        self._initialize_data(data, data_path)
        self._process_data()
        self._set_variables(variable_names, variable_units)
        self._analyze_data()
        self._set_functions(functions)
        
        if self._generate_fuzzy:
            self._set_prompts_for_fuzzy()
        else:
            self._set_prompts()
        
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
        
        if self._generate_fuzzy:
            idea = self._bring_fuzzy_to_life(idea)
        
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
                    score = -2.0
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
                f"拟设：{idea}\n"
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
        
        report_lines.append("")
        
        y_min = y_data.min()
        y_max = y_data.max()
        report_lines.append(f"输出范围: [{y_min:.4f}, {y_max:.4f}]")
  
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
        
        report_lines = []
        
        mse = np.mean(residuals ** 2)
        mae = np.mean(abs_residuals)
        max_error = np.max(abs_residuals)
        
        report_lines.append(f"拟合误差: MSE={mse:.3e}, MAE={mae:.3e}, 最大误差={max_error:.3e}")
        
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
            assert self._output_name is not None
            if self._variable_descreption is None:
                self._variable_descreption = {}
            
            if list(self._variable_descreption.keys()) != self._variables:
                for var in self._variables:
                    if var not in self._variable_descreption:
                        self._variable_descreption[var] = "未知物理量"
            if self._output_descreption is None:
                self._output_descreption = "未知物理量"

            variables_info = (
                f"另外， {', '.join(self._variables)} 是物理量，"
                f"这些物理量的单位分别为 {', '.join(self._variable_units)} ，"
                f"这些物理量的含义为： {', '.join([var +':'+self._variable_descreption[var] for var in self._variables])} 。\n"
                "为了简化问题，所有的可优化参数 param 都是无量纲（单位为 1 ）的；这些经验参数越少越好。\n"
                f"你要用这些物理量和经验参数 param 构造一个表达式来描述一个单位为 {self._output_unit} 的物理量{self._output_name}"
                f"这个物理量的含义为：{self._output_descreption}，请务必确保表达式的量纲的正确性。"
                f"为此，必要时你可能需要构造一些无量纲量来进行复杂的函数运算，并随后将量纲匹配到目标输出{self._output_name}上。"
            )

            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info

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
            "分析以上示例时，如果你发现一些特征，可能可以以启发式的方法重构参数，例如，"
            "如果某个参数的最优值始终非常接近零，说明它在该任务中作用不大，"
            "你可以考虑在新的拟设中去除这一参数，以提高结构紧凑性与有效性。\n"
            "同样的，如果某两个参数相近，某个参数列接近一个函数的泰勒展开系数等等，都可能能够帮助你注意到或者猜出更合理的函数形式"
            f"{variables_info}"
            "其次，如果遇到非独立参数（例如 param1 * (param2 * x + param3)），"
            "请将其整理为 param1 * x + param2 的形式，让各个参数尽量独立。\n"
            "最后，注意拟设格式规则意味着你需要显式写出倍数和幂率\n"
            "比如你必须将3*x写作(x+x+x)，或将y**2写作(y*y)，这样才能确保函数被正确读取。\n"
            "拟设格式规则同样要求你不得使用1，0，-1等数字，这就意味着你必须将1写作(x/x)，x**-1写作param1/(param1*x)等形式。\n"
            "现在，请你开始生成 expression 表达式。"
            "请记住，直接输出合法的表达式字符串，不要包含解释、注释或额外内容，方便我们接入自动任务流。"
        )
        
    
    def _set_prompts_for_fuzzy(
        self,
    )-> None:
        
        """
        configure system_prompt, prologue_section, epilogue_section for IdeaSearcher (fuzzy version)
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
            assert self._output_name is not None
            
            if self._variable_descreption is None:
                self._variable_descreption = {}

            if self._variable_descreption.keys() != self._variables:
                for var in self._variables:
                    if var not in self._variable_descreption:
                        self._variable_descreption[var] = "未知物理量"
                        
            if self._output_descreption is None:
                self._output_descreption = "未知物理量"

            variables_info = (
                f"另外， {', '.join(self._variables)} 是物理量，"
                f"这些物理量的单位分别为 {', '.join(self._variable_units)} ，"
                f"这些物理量的含义为： {', '.join([var +':' + self._variable_descreption[var] for var in self._variables])} 。\n"
                "为了简化问题，所有的可优化参数 param 都是无量纲（单位为 1 ）的；这些经验参数越少越好。\n"
                f"你要用这些物理量和经验参数 param 构造一个表达式来描述一个单位为 {self._output_unit} 的物理量{self._output_name}"
                f"这个物理量的含义为：{self._output_descreption}，请务必确保表达式的量纲的正确性。"
                f"为此，必要时你可能需要构造一些无量纲量来进行复杂的函数运算，并随后将量纲匹配到目标输出{self._output_name}上。"
            )
            
            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info
                
        else:
            variables_info = (
                f"另外，由于 {', '.join(self._variables)} 实际上是物理量，"
                f"我们鼓励你在将 {', '.join(self._variables)} 设为函数自变量前设置一个参数与之相乘。\n"
            )

        self.system_prompt: str = (
            "你是一位富有创造力的物理学家和数据科学家，擅长从数据中发现潜在的规律，并用自然语言和数学公式清晰地表达出来。"
            "你的任务是分析给定的数据和背景信息，提出一个关于其背后物理机制的连贯的理论分析，并给出一个最能描述该理论的数学公式。"
        )

        self.prologue_section: str = (
            "首先，请你理解本任务的背景信息：\n"
            "我们正在尝试寻找一个数学表达式来拟合一组实验数据。\n"
            f"可用的自变量为: [{prologue_section_variable_string}]\n"
            f"可用的函数为: [{prologue_section_function_string}]\n"
            f"其次，请查看待拟合函数的一份简短报告：\n{self._data_info}\n"
            "然后，为了帮助你理解风格和注意事项，我们先提供一些已有的 expression 示例，请仔细观察它们的结构与潜在问题，总结经验教训后再开始生成，一定不许重复示例中的内容：\n"
        )

        self.epilogue_section: str = (
            "分析以上示例时，如果你发现一些特征，可能可以以启发式的方法重构参数，例如，"
            "如果某个参数的最优值始终非常接近零，说明它在该任务中作用不大。\n"
            "同样的，如果某两个参数相近，某个参数列接近一个函数的泰勒展开系数等等，都可能能够帮助你注意到或者猜出更合理的函数形式。\n"
            f"{variables_info}\n"
            "现在，请你开始撰写你的分析理论。\n"
            "你的回答应该包含两部分：\n"
            "1.  **理论分析**：一段连贯的文字，阐述你对数据背后规律的洞察和推理过程。\n"
            "2.  **数学公式**：在分析的最后，请给出一个你认为最合适的数学公式。请使用标准的数学排版，并用 `<final_result>` 将其包裹起来，例如 `<final_result> y = a * sin(b * x) + c </final_result>`。\n"
            "请确保你的分析富有洞察力，公式有合理的物理含义，且不与示例中内容重复。"
        )


    def _bring_fuzzy_to_life(
        self, 
        fuzzy: str, 
    )-> str:

        prologue_section_variable_string = ", ".join(
            [f'"{variable}"' for variable in self._variables]
        )
        prologue_section_function_string = ", ".join(
            [f'"{function}"' for function in self._functions]
        )

        system_prompt = (
            "你是一个严格遵循指令的代码转换器。"
            "你的任务是将一个包含自然语言和标准数学公式的理论描述，转换成一个严格符合特定语法的表达式字符串。"
        )
        
        formula_part = ""
        final_result_pattern = r'<final_result>(.*?)</final_result>'
        matches = re.findall(final_result_pattern, fuzzy, re.DOTALL)
        
        if matches:
            formula_part = matches[-1].strip()
        else:
            formula_part = '[未找到最终公式，请参考上面的理论描述结果构造表达式]'
    
        
        user_prompt = (
            f"请将以下理论描述中的数学公式转换成严格的拟设表达式。\n"
            f"公式所来自的理论（仅参考）：\n---\n{fuzzy}\n---\n\n"
            f"请严格遵守以下格式规则：\n"
            f"1.  **拟设格式**: 完整格式为：\n{ansatz_docstring}\n"
            f"2.  **可用变量**: `variables = [{prologue_section_variable_string}]`\n"
            f"3.  **可用函数**: `functions = [{prologue_section_function_string}]`\n"
            "    这些是你在拟设表达式中唯一可以使用的变量和函数。\n"
            "4.  **无数字**: 不得使用任何数字（如 1, 0, -1, 3.14）。\n"
            "    -   `1` 必须写成 `(x/x)` (其中 `x` 是任意变量)。\n"
            "    -   `-1` 必须写成 `(x-x)-y/y` 的形式。\n"
            "    -   其他数字必须通过变量和参数构造，例如 `2` 可以是 `param1/param2` 并在后续优化，或者写成 `(x+x)/x`。\n"
            "5.  **显式乘幂**: 必须显式写出倍数和幂率。\n"
            "    -   `3*x` 必须写成 `(x+x+x)`。\n"
            "    -   `y**2` 必须写成 `(y*y)`。\n"
            "    -   `y**-1` 必须写成 `(x/x)/y` 的形式。\n"
            "6.  **独立参数**: 避免非独立参数。例如，`param1 * (param2 * x + param3)` 应整理为 `param1 * x + param2` 的形式（通过重命名参数）。\n"
            "7.  **至少一个参数**： 确保表达式中至少包含一个参数 `param1`，以便后续优化。\n"
            "7.  **输出**: 只输出最终的表达式字符串，不要包含任何解释、注释或额外内容。以便于我们接入自动处理信息流。\n\n"
            f"例如，如果给你的内容是 `y = 2*x + 1（其中x为自变量，y为因变量）`，你应该输出 `(x+x) + param1 * (x/x)`或`param1 * x + (x/x)`。\n"
            f"现在，请转换这个公式：`{formula_part}`"
        )
        
        llm_response = get_answer(
            prompt = user_prompt,
            system_prompt = system_prompt,
            model_name = "Qwen_Max",
            model_temperature = 0.0,
        )
        
        llm_answer = re.sub(r'[`\'\"<>]', '', llm_response)

        return llm_answer


    def _set_naive_linear_idea(
        self,
    )-> None:
        
        self._naive_linear_idea: str = " + ".join([
            f"param{i + 1} * {variable}"
            for i, variable in enumerate(self._variables)
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
            f"{variable}": self._x_rescaled[:, i]
            for i, variable in enumerate(self._variables)
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
        
        natural_param_range = (-10.0, 10.0)
                
        best_params, best_metric_value = ansatz.apply_to(
            numeric_ansatz_user = numeric_ansatz_user,
            param_ranges = [natural_param_range] * ansatz_param_num,
            trial_num = 100,
            method = "L-BFGS-B",
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