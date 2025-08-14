import os
import numexpr
import numpy as np
from numpy import ndarray
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import Callable
from typing import Optional
from numpy.random import default_rng
from pywheels.math_funcs import chi_squared
from pywheels.math_funcs import mean_squared_error
from pywheels.blueprints.ansatz import Ansatz
from pywheels.blueprints.ansatz import ansatz_docstring
from .i18n import translate


__all__ = [
    "IdeaSearchFitter",
]


class IdeaSearchFitter:
    
    # ------------------------- IdeaSearchFitter初始化 -------------------------
    
    def __init__(
        self,
        data: Optional[Dict[str, ndarray]] = None,
        data_path: Optional[str] = None,
        functions: List[str] = ["sin", "cos", "exp", "log", "sqrt", "power"],
        baseline_metric_value: Optional[float] = None, # metric value corresponding to score 0.0
        good_metric_value: Optional[float] = None, # metric value corresponding to score 80.0
        metric_mapping: Literal["linear", "logarithm"] = "linear",
        enable_mutation: bool = False,
        enable_crossover: bool = False,
        seed: Optional[int] = None,
    )-> None:
        
        self._random_generator = default_rng(seed)

        self._initialize_data(
            data = data,
            data_path = data_path,
        )
        
        self._set_variables()
        
        self._set_functions(
            functions = functions,
        )
        
        self._set_prompts()
        
        self._build_metric_mapper(
            baseline_metric_value = baseline_metric_value,
            good_metric_value = good_metric_value,
            metric_mapping = metric_mapping,
        )
        
        self._set_initial_ideas()
        
        # hijack action `mutation_func` and `crossover_func` when disabled
        if not enable_mutation: self.mutation_func = None # type: ignore
        if not enable_crossover: self.crossover_func = None # type: ignore
    
    # ----------------------------- 外部动作 -----------------------------

    def evaluate_func(
        self,
        idea: str
    )-> Tuple[float, Optional[str]]:

        try:

            best_params, best_metric_value = self._optimize_idea_under_metric(
                idea = idea,
            )
            
            best_params_msg = ""
            
            for index, best_param in enumerate(best_params):
                best_params_msg += f"  param{index+1}: {best_param:.4g}"
                
            score = self._metric_mapper(best_metric_value)
            info = (
                f"得分：{score:.2f}\n"
                f"{self._metric_type}：{best_metric_value:.8g}\n"
                f"最优参数：{best_params_msg}"
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
    
    # ----------------------------- 内部动作 -----------------------------
    
    def _initialize_data(
        self,
        data: Optional[Dict[str, ndarray]] = None,
        data_path: Optional[str] = None,
    )-> None:
        
        if (data is None and data_path is None) or \
            (data is not None and data_path is not None):
                
            raise ValueError(translate(
                "【IdeaSearchFitter】初始化时出错：应在 data 与 data_path 间选择一个参数传入！"
            ))
            
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
            
        fit_with_errors: bool = (self._error is not None)
        
        self._metric_type = "chi squared" if fit_with_errors else "mean square error"
        self._metric_func = chi_squared if fit_with_errors else mean_squared_error
        self._metric_input = {"ground_truth_data": self._y} if fit_with_errors else \
            {"ground_truth_data": self._y, "errors": self._error}
        
        
    def _set_variables(
        self,
    )-> None:
        
        self._input_dim: int = self._x.shape[1]
        
        self._variables: List[str] = [
            f"x{i + 1}" for i in range(self._input_dim)
        ]
        
        self._numexpr_local_dict: Dict[str, ndarray] = {
            f"x{i + 1}": self._x[:, i]
            for i in range(self._input_dim)
        }
        
        self._naive_linear_idea: str = " + ".join([
            f"param{i + 1} * x{i + 1}"
            for i in range(self._input_dim)
        ])
        
        
    def _set_functions(
        self,
        functions: List[str],
    )-> None:
        
        self._functions: List[str] = functions
    
    
    def _set_prompts(
        self,
    )-> None:
        
        prologue_section_variable_string = ", ".join(
            [f'"{variable}"' for variable in self._variables]
        )
        
        prologue_section_function_string = ", ".join(
            [f'"{function}"' for function in self._functions]
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
            "为了帮助你理解风格和注意事项，我们先提供一些已有的 expression 示例，请仔细观察它们的结构与潜在问题，总结经验教训后再开始生成：\n"
        )
        
        self.epilogue_section: str = (
            "分析以上示例时，如果你发现某个参数的最优值始终非常接近零，说明它在该任务中作用不大，"
            "你可以考虑在新的拟设中去除这一参数，以提高结构紧凑性与有效性。\n"
            f"另外，由于 {', '.join(self._variables)} 实际上是物理量，"
            f"我们鼓励你在 {', '.join(self._variables)} 前设置一个参数与之相乘，然后再代入函数。\n"
            "最后，如果遇到非独立参数（例如 param1 * (param2 * x + param3)），"
            "请将其整理为 param1 * x + param2 的形式，让各个参数尽量独立。\n"
            "现在，请你开始生成 expression 表达式。"
            "请记住，直接输出合法的表达式字符串，不要包含解释、注释或额外内容，方便我们接入自动任务流。"
        )
        
        
    def _build_metric_mapper(
        self,
        baseline_metric_value: Optional[float],
        good_metric_value: Optional[float],
        metric_mapping: Literal["linear", "logarithm"],
    )-> None:
        
        if baseline_metric_value is None:
            
            _, baseline = self._optimize_idea_under_metric(
                idea = self._naive_linear_idea
            )
            
        else:
            baseline = baseline_metric_value
            
        if good_metric_value is None:
            good = baseline / 100
            
        else:
            good = good_metric_value
            
        good_raw_score = 80.0
        self._metric_mapper: Callable[[float], float] = \
            lambda metric_value: \
            min(100.0, 
                max(
                    good_raw_score * (np.log(baseline / metric_value)) / (np.log(baseline / good)) \
                    if metric_mapping == "logarithm" else\
                    good_raw_score * ((baseline - metric_value) / (baseline - good)), 
                    0.0
                )
            )
            
            
    def _optimize_idea_under_metric(
        self,
        idea: str,
    )-> Tuple[List[float], float]:
        
        ansatz = Ansatz(
            expression = idea,
            variables = self._variables,
            functions = self._functions,
        )

        def numeric_ansatz_user(
            numeric_ansatz: str
        )-> float:
            
            y_pred = numexpr.evaluate(
                ex = numeric_ansatz, 
                local_dict = self._numexpr_local_dict,
            )
                
            metric_value = self._metric_func(
                predicted_data = y_pred,
                **self._metric_input
            )
            
            return metric_value
        
        natural_param_range = (-10.0, 10.0)
                
        best_params, best_metric_value = ansatz.apply_to(
            numeric_ansatz_user = numeric_ansatz_user,
            param_ranges = [natural_param_range] * ansatz.get_param_num(),
            trial_num = 5,
            mode = "optimize",
        )
        
        return best_params, best_metric_value
        
    
    def _set_initial_ideas(
        self,
    )-> None:
        
        self.initial_ideas: List[str] = [self._naive_linear_idea]