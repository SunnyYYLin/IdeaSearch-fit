from .utils import *
from .unit_validator import *


__all__ = [
    "IdeaSearchFitter",
]


_numexpr_supported_functions: List[str] = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", 
    "cosh", "tanh", "log", "log10", "exp", "square", "sqrt", "abs", 
]


_default_functions: List[str] = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "tanh", "log", "log10", "exp", "square", "sqrt", "abs",
]


class IdeaSearchFitter:
    
    # ------------------------- IdeaSearchFitter初始化 -------------------------
    
    def __init__(
        self,
        data: Optional[Dict[str, ndarray]] = None,
        data_path: Optional[str] = None,
        existing_fit: str = "0.0",
        functions: List[str] = _default_functions.copy(),
        constant_whitelist: List[str] = [],
        constant_map: Dict[str, float] = {"pi": np.pi},
        perform_unit_validation: bool = False,
        input_description: Optional[str] = None,
        variable_description: Optional[Dict[str, str]] = None,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
        output_description: Optional[str] = None,
        output_name: Optional[str] = None,
        output_unit: Optional[str] = None,
        generate_fuzzy: bool = True,
        fuzzy_translator: Optional[str] = None,
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
            generate_fuzzy = generate_fuzzy,
            fuzzy_translator = fuzzy_translator,
        )
        
        self._random_generator = default_rng(seed)
        
        self._existing_fit: str = existing_fit
        self._perform_unit_validation: bool = perform_unit_validation
        self._auto_rescale: bool = auto_rescale
        self._adjust_degrees_of_freedom: bool = adjust_degrees_of_freedom
        self._generate_fuzzy: bool = generate_fuzzy
        self._fuzzy_translator = fuzzy_translator
        
        self._output_unit: Optional[str] = output_unit
        self._output_name: Optional[str] = output_name
        self._variable_description: Optional[Dict[str,str]] = variable_description
        self._output_description: Optional[str] = output_description
        self._input_description: Optional[str] = input_description

        self._initialize_data(data, data_path)
        self._process_data()
        self._set_variables(variable_names, variable_units)
        self._analyze_data()
        self._set_functions(functions)
        self._constant_whitelist = constant_whitelist
        self._constant_map = constant_map
        
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

    @lru_cache(maxsize = 2048)
    def evaluate_func(
        self,
        idea: str
    )-> Tuple[float, Optional[str]]:
        
        try:
            
            ansatz = Ansatz(
                expression = idea,
                variables = self._variables,
                functions = self._functions,
                constant_whitelist = self._constant_whitelist,
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
        
        
    def postprocess_func(
        self, 
        raw_response: str, 
    )-> str:
        
        if not self._generate_fuzzy: return raw_response
        assert self._fuzzy_translator is not None
        fuzzy = raw_response

        prologue_section_variable_string = ", ".join(
            [f'"{variable}"' for variable in self._variables]
        )
        prologue_section_function_string = ", ".join(
            [f'"{function}"' for function in self._functions]
        )
        prologue_section_number_string = ", ".join(
            [f'"{number}"' for number in self._constant_whitelist]
        )

        system_prompt = (
            "You are a code translator that strictly follows instructions."
            "Your task is to convert a theoretical description, which includes natural language and standard mathematical formulas, into an expression string that strictly conforms to a specific syntax."
        )
        
        formula_part = ""
        final_result_pattern = r'<final_result>(.*?)</final_result>'
        matches = re.findall(final_result_pattern, fuzzy, re.DOTALL)
        
        if matches:
            formula_part = matches[-1].strip()
        else:
            formula_part = "[Final formula not found. Please construct the expression based on the theoretical description above.]"
        
        user_prompt = (
            f"Please convert the mathematical formula from the following theoretical description into a strict ansatz expression.\n"
            f"The theory from which the formula originates (for reference only):\n---\n{fuzzy}\n---\n\n"
            f"Please strictly adhere to the following formatting rules:\n"
            f"1.  **Ansatz Format**: The complete format is:\n{ansatz_docstring}\n"
            f"2.  **Available Variables**: `variables = [{prologue_section_variable_string}]`\n"
            f"3.  **Available Functions**: `functions = [{prologue_section_function_string}]`\n"
            f"4.  **Available Constants**: `available constants = [{prologue_section_number_string}]`\n"
            "    These are the only variables, functions, and constants you are allowed to use in the ansatz expression.\n"
            "4.  **Legal Constants**: Do not use any constants outside of the available list (e.g., 0.3, 1.7, 9.7, 11).\n"
            "    -   `3` can be written as `(x+x+x)/x` (where `x` is any variable).\n"
            "    -   Numbers can also be constructed from variables and parameters. For example, `-4` could be `param1*param2` for later optimization, or written as `(-2-2)`.\n"
            "5.  **Explicit Powers and Multiples**: You must explicitly write out multiples and powers, unless they can be constructed with available constants or functions.\n"
            "    -   `3*x` must be written as `(x+x+x)`.\n"
            "    -   `y**2` can be written as `(y**2)` or `(square(y))`.\n"
            "6.  **Independent Parameters**: Avoid non-independent parameters. For example, `param1 * (param2 * x + param3)` should be refactored into a form like `param1 * x + param2` (by renaming parameters).\n"
            "7.  **Parameter Range**: There are no strict limits on parameter ranges. However, since initial parameter values are sampled from (-10, 10), you may adjust the parameter's form (e.g., writing `param1` as `2**param1`) to avoid excessively large values and facilitate optimization.\n"
            "8.  **At Least One Parameter**: Ensure the expression contains at least one parameter, `param1`, to enable subsequent optimization.\n"
            "9.  **Output**: Output only the final expression string. Do not include any explanations, comments, or extra content, to facilitate its integration into our automated workflow.\n\n"
            f"For example, if the input is `y = 2*x + 3` (where x is the independent variable and y is the dependent variable), you should output `2*x + param1` or `2 * x + 2 + param1`.\n"
            f"Now, please convert this formula: `{formula_part}`"
        )
        
        llm_response = get_answer(
            prompt = user_prompt,
            system_prompt = system_prompt,
            model = self._fuzzy_translator,
            temperature = 0.0,
        )
        
        llm_answer = re.sub(r'[`\'\"<>]', '', llm_response)

        return llm_answer
        
    
    def mutation_func(
        self,
        idea: str,
    )-> str:

        ansatz = Ansatz(
            expression = idea,
            variables = self._variables,
            functions = self._functions,
            constant_whitelist = self._constant_whitelist,
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
                constant_whitelist = self._constant_whitelist,
            )
            
            ansatz2 = Ansatz(
                expression = parent2,
                variables = self._variables,
                functions = self._functions,
                constant_whitelist = self._constant_whitelist,
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
        perform_unit_validation: bool,
        variable_names: Optional[List[str]],
        variable_units: Optional[List[str]],
        output_unit: Optional[str],
        generate_fuzzy: bool,
        fuzzy_translator: Optional[str],
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
            
        if generate_fuzzy and fuzzy_translator is None:
            raise ValueError(translate(
                "【IdeaSearchFitter】 初始化时出错：生成中间模糊报告时必须设置模糊报告转译模型 fuzzy_translator！"
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
            if self._variable_description is None:
                self._variable_description = {}
            
            if list(self._variable_description.keys()) != self._variables:
                for var in self._variables:
                    if var not in self._variable_description:
                        self._variable_description[var] = "Unknown physical quantity"
            if self._output_description is None:
                self._output_description = "Unknown physical quantity"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are physical quantities, "
                f"with respective units of {', '.join(self._variable_units)}, "
                f"and their meanings are as follows: {', '.join([var +':'+self._variable_description[var] for var in self._variables])}.\n"
                "To simplify the problem, all optimizable parameters, denoted as `param`, are dimensionless (unit of 1). The number of these empirical parameters should be minimized.\n"
                f"Your task is to construct an expression using these physical quantities and empirical parameters to describe a physical quantity `{self._output_name}` with units of `{self._output_unit}`."
                f"The meaning of this quantity is: {self._output_description}. It is crucial to ensure the dimensional correctness of the expression."
                f"To achieve this, you may need to construct dimensionless quantities for complex function operations, and subsequently match the dimensions to the target output `{self._output_name}`."
            )

            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info

        else:
        
            variables_info = (
                f"Additionally, since {', '.join(self._variables)} are physical quantities, "
                f"we encourage you to multiply them by a parameter before using them as function arguments.\n"
            )
        
        self.system_prompt: str = (
            "You are an experimental scientist who strictly follows instructions, adept at generating innovative and correctly formatted new ansatz expressions based on existing structures."
            "You will observe the performance of existing ansaetze, summarize patterns, and generate new `expression` strings that are both valid and potentially superior."
        )
        
        self.prologue_section: str = (
            f"First, please review the following ansatz formatting rules:\n{ansatz_docstring}\n"
            "In this task, you only need to generate the `expression` part. The `variables` and `functions` are fixed as follows:\n"
            f"variables = [{prologue_section_variable_string}]\n"
            f"functions = [{prologue_section_function_string}]\n"
            "Note that these are the only variables and functions you are allowed to use.\n"
            f"Second, please review a brief report on the function to be fitted:\n{self._data_info}\n"
            "Next, to help you understand the required style and important considerations, we provide some examples of existing `expression`s. Please carefully observe their structure and potential issues, learn from them, and then begin your generation:\n"
        )
        
        self.epilogue_section: str = (
            "When analyzing the examples, if you identify certain features, you might heuristically restructure the parameters. For example, "
            "if the optimal value of a parameter is consistently close to zero, it suggests it has little impact on the task. "
            "You might consider removing this parameter in your new ansatz to improve structural compactness and effectiveness.\n"
            "Similarly, if two parameters are close in value, or if a series of parameters resembles the Taylor expansion coefficients of a function, these observations might help you notice or deduce a more reasonable functional form."
            f"{variables_info}"
            "Furthermore, if you encounter non-independent parameters (e.g., `param1 * (param2 * x + param3)`), "
            "please refactor them into a form like `param1 * x + param2` to make each parameter as independent as possible.\n"
            "Finally, note that the ansatz formatting rules require you to explicitly write out multiplications and powers.\n"
            "For instance, you must write `3*x` as `(x+x+x)` and `y**2` as `(y*y)` to ensure the function is parsed correctly.\n"
            "The rules also forbid the use of numeric literals like 1, 0, or -1. This means you must represent 1 as `(x/x)`, `x**-1` as `param1/(param1*x)`, or in similar forms.\n"
            "Now, please begin generating the `expression` string."
            "Remember to output only the valid expression string, without any explanations, comments, or extra content, to facilitate its integration into our automated workflow."
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
                        self._variable_descreption[var] = "Unknown quantity"
                        
            if self._output_descreption is None:
                self._output_descreption = "Unknown quantity"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are physical quantities, "
                f"with respective units of {', '.join(self._variable_units)}, "
                f"and their meanings are as follows: {', '.join([var +':' + self._variable_descreption[var] for var in self._variables])}.\n"
                "To simplify the problem, all optimizable parameters, denoted as `param`, are dimensionless (unit of 1). The number of these empirical parameters should be minimized.\n"
                f"Your task is to construct an expression using these physical quantities and empirical parameters to describe a physical quantity `{self._output_name}` with units of `{self._output_unit}`."
                f"The meaning of this quantity is: {self._output_descreption}. It is crucial to ensure the dimensional correctness of the expression."
                f"To achieve this, you may need to construct dimensionless quantities for complex function operations, and subsequently match the dimensions to the target output `{self._output_name}`."
            )
            
            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info
                
        else:
            assert self._output_name is not None
            
            if self._variable_descreption is None:
                self._variable_descreption = {}

            if self._variable_descreption.keys() != self._variables:
                for var in self._variables:
                    if var not in self._variable_descreption:
                        self._variable_descreption[var] = "[Undefined Meaning]"
                        
            if self._output_descreption is None:
                self._output_descreption = "[Undefined Meaning]"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are the input quantities, "
                f"and their meanings are as follows: {', '.join([var +':' + self._variable_descreption[var] for var in self._variables])}.\n"
                "To simplify the problem, all optimizable empirical parameters, denoted as `param`, should be as few as possible and preferably dimensionless.\n"
                f"Your task is to construct an expression using these quantities and empirical parameters to describe a target quantity: `{self._output_name}`."
                f"The meaning of this target quantity is: {self._output_descreption}."
            )
            
            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info

        self.system_prompt: str = (
            "You are a creative data scientist, skilled at discovering underlying patterns in data and articulating them clearly through natural language and mathematical formulas."
            "Your task is to analyze the given data and background information, propose a coherent theoretical analysis of the underlying mechanism, and provide the mathematical formula that best describes this theory."
        )

        self.prologue_section: str = (
            "First, please understand the background information for this task:\n"
            "We are attempting to find a mathematical expression to fit a set of experimental data.\n"
            f"The available independent variables are: [{prologue_section_variable_string}]\n"
            f"The available functions are: [{prologue_section_function_string}]\n"
            f"Second, please review a brief report on the function to be fitted:\n{self._data_info}\n"
            "Next, to help you understand the style and important considerations, we provide some examples of existing expressions. Please carefully observe their structure and potential issues, learn from them, and then begin your generation. You must not repeat the content from the examples:\n"
        )

        self.epilogue_section: str = (
            "When analyzing the examples, if you identify certain features, you might heuristically restructure the parameters. For example, "
            "if the optimal value of a parameter is consistently close to zero, it suggests it has little impact on the task.\n"
            "Similarly, if two parameters are close in value, or if a series of parameters resembles the Taylor expansion coefficients of a function, these observations might help you notice or deduce a more reasonable functional form.\n"
            f"{variables_info}\n"
            "Now, please begin writing your theoretical analysis.\n"
            "Your response should consist of two parts:\n"
            "1.  **Theoretical Analysis**: A coherent text explaining your insights and reasoning process regarding the underlying patterns in the data.\n"
            "2.  **Mathematical Formula**: At the end of your analysis, provide what you believe is the most suitable mathematical formula. Please use standard mathematical typesetting and enclose it in `<final_result>` tags, for example: `<final_result> y = a * sin(b * x) + c </final_result>`.\n"
            "Please ensure your analysis is insightful, the formula has a reasonable theoretical meaning, and its content does not overlap with the examples provided."
        )


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
        
        variable_local_dict = {
            f"{variable}": self._x_rescaled[:, i]
            for i, variable in enumerate(self._variables)
        }
        self._numexpr_local_dict = {
            **variable_local_dict,
            **self._constant_map,
        }
            
            
    def _get_idea_optimal_result(
        self,
        idea: str,
    )-> Tuple[List[float], float]:
        
        ansatz = Ansatz(
            expression = idea,
            variables = self._variables,
            functions = self._functions,
            constant_whitelist = self._constant_whitelist,
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
                constant_whitelist = self._constant_whitelist,
            )
            
            self._best_fit = ansatz.reduce_to_numeric_ansatz(best_params)