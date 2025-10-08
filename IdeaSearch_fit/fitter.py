from .utils import *
from .unit_validator import *
from .pareto_frontier import *


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
        result_path: str,
        data: Optional[Dict[str, ndarray]] = None,
        data_path: Optional[str] = None,
        functions: List[str] = deepcopy(_default_functions),
        constant_whitelist: List[str] = [],
        constant_map: Dict[str, float] = {"pi": np.pi},
        perform_unit_validation: bool = False,
        input_description: Optional[str] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
        variable_names: Optional[List[str]] = None,
        variable_units: Optional[List[str]] = None,
        output_description: Optional[str] = None,
        output_name: Optional[str] = None,
        output_unit: Optional[str] = None,
        auto_polish: bool = True,
        auto_polisher: Optional[str] = None, 
        generate_fuzzy: bool = True,
        fuzzy_translator: Optional[str] = None,
        baseline_metric_value: Optional[float] = None, # metric value corresponding to score 20.0
        good_metric_value: Optional[float] = None, # metric value corresponding to score 80.0
        metric_mapping: Literal["linear", "logarithm"] = "linear",
        enable_mutation: bool = False,
        enable_crossover: bool = False,
        optimization_method: Literal["L-BFGS-B", "differential-evolution"] = "L-BFGS-B",
        optimization_trial_num: int = 100,
        seed: Optional[int] = None,
    )-> None:
        
        # Under development and testing
        auto_rescale = False
        existing_fit = "0.0"
        adjust_degrees_of_freedom = False
        
        self._preflight_check(
            result_path = result_path,
            data = data,
            data_path = data_path,
            perform_unit_validation = perform_unit_validation,
            variable_names = variable_names,
            variable_units = variable_units,
            output_unit = output_unit,
        )
        
        default_model = get_available_models()[0]
        if auto_polisher is None: auto_polisher = default_model
        if fuzzy_translator is None: fuzzy_translator = default_model
        
        self._random_generator = default_rng(seed)
        
        self._optimization_method: Literal["L-BFGS-B", "differential-evolution"] = optimization_method
        self._optimization_trial_num = optimization_trial_num
        
        self._result_path = result_path
        self._pareto_report_path = f"{result_path}{seperator}pareto_report.txt"
        self._pareto_data_path = f"{result_path}{seperator}pareto_data.json"
        self._pareto_frontier: Dict[int, Dict] = {}
        self._pareto_frontier_lock = Lock()
        
        self._existing_fit: str = existing_fit
        self._perform_unit_validation: bool = perform_unit_validation
        self._auto_rescale: bool = auto_rescale
        self._adjust_degrees_of_freedom: bool = adjust_degrees_of_freedom
        
        self._generate_fuzzy: bool = generate_fuzzy
        self._fuzzy_translator = fuzzy_translator
        self._idea_to_fuzzy: Dict[str, str] = {}
        
        self._output_unit: Optional[str] = output_unit
        self._output_name: Optional[str] = output_name
        self._variable_descriptions: Optional[Dict[str,str]] = variable_descriptions
        self._output_description: Optional[str] = output_description
        self._input_description: Optional[str] = input_description
        self._auto_polisher = auto_polisher

        self._initialize_data(data, data_path)
        self._process_data()
        self._set_variables(variable_names, variable_units)
        
        if auto_polish: self._auto_polish()
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
            best_numeric_ansatz = ansatz.reduce_to_numeric_ansatz(best_params)
            
            y_pred_remainder_rescaled = numexpr.evaluate(
                ex = best_numeric_ansatz, 
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
            info_dict = {
                "ansatz": idea,
                "score": score,
                f"{metric_type}": true_metric_value,
                "best_parameters": best_params_msg,
                "best_numeric_ansatz": best_numeric_ansatz,
                "best_numeric_ansatz_residual_report": residual_report,
                "created_at": get_time_stamp(show_minute=True, show_second=True),
            }
            if self._generate_fuzzy:
                info_dict["fuzzy_enlightor"] = self._idea_to_fuzzy.get(idea, "NOT AVAILABLE")
            info = serialize_json(info_dict)
            
            self._update_pareto_frontier(
                numeric_ansatz = best_numeric_ansatz,
                metric_value = best_metric_value,
                info_dict = info_dict,
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
        
        idea = re.sub(r'[`\'\"<>]', '', llm_response)
        self._idea_to_fuzzy[idea] = raw_response

        return idea
        
    
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
    
    
    def get_pareto_frontier(
        self,
    )-> Dict[int, Dict]:
        
        return deepcopy(self._pareto_frontier)
    
    # ----------------------------- 内部动作 -----------------------------
    
    def _preflight_check(
        self,
        result_path: str,
        data: Optional[Dict[str, ndarray]],
        data_path: Optional[str],
        perform_unit_validation: bool,
        variable_names: Optional[List[str]],
        variable_units: Optional[List[str]],
        output_unit: Optional[str],
    )-> None:
        
        if not os.path.isdir(result_path):
            raise ValueError(translate(
                "【IdeaSearchFitter】初始化时出错：result_path 应指向一存在的文件夹，用于存放帕累托前沿等拟合结果！"
            ))
        
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
            
            
    def _auto_polish(
        self,
    )-> None:
        
        # Skip auto_polish if output_name is not provided
        if self._output_name is None:
            return
        
        if all(item is not None for item in [
            self._input_description,
            self._variable_descriptions,
            self._output_description,
        ]): return
        
        if self._variable_units is None:
            input_variable_string = ", ".join(self._variables)
        else:
            input_variable_string = ", ".join(f"{variable} ({unit})" for variable, unit in zip(
                self._variables,
                self._variable_units,
            ))

        if self._output_unit is None:
            output_variable_string = self._output_name
        else:
            output_variable_string = f"{self._output_name} ({self._output_unit})"
        
        system_prompt = "You are a domain expert skilled at inferring the physical or conceptual meaning of variables from their names and associated units."
        
        expected_keys_string = ", ".join(f'"{key}"' for key in self._variables)
        
        prompt = f"""
As a domain expert, your task is to provide clear, concise descriptions for the following variables based on the provided information.

### Variables
- **Input Features**: {input_variable_string}
- **Target Variable**: {output_variable_string}

### Instructions
1.  **Analyze**: Infer the likely meaning of each variable based on its name and unit.
2.  **Describe**: Provide a concise, one to two-sentence explanation for each variable. If a variable's meaning is ambiguous, state that it is unclear or list the possible interpretations instead of inventing a definition.
3.  **Format**: Your response MUST be a single, valid JSON object wrapped in a markdown code block. For example:
    ```json
    {{
    "input_description": "...",
    "variable_descriptions": {{...}},
    "output_description": "..."
    }}
    ```
    - `input_description`: A string providing a brief, high-level overview of the dataset or physical system.
    - `variable_descriptions`: A dictionary where each key is a feature variable name and the value is its description. This dictionary **MUST** contain entries for exactly the following keys: {expected_keys_string}.
    - `output_description`: A string describing the meaning of the target variable.

Ensure your final output is only the markdown block containing the JSON, without any other surrounding text or explanations.
"""

        polished_results: Dict[str, Any] = {}
        def check_and_accept(
            response: str,
        )-> bool:
            nonlocal polished_results
            try:
                pattern = r"```json\s*\n(.*?)\n```"
                json_match = re.search(pattern, response, re.DOTALL)
                assert json_match
                json_string = json_match.group(1).strip()
                
                data = json.loads(json_string)
                if not isinstance(data, dict): return False

                required_keys = {"input_description", "variable_descriptions", "output_description"}
                if not required_keys.issubset(data.keys()): return False

                if not all(isinstance(data[key], str) for key in ["input_description", "output_description"]):
                    return False
                
                if not isinstance(data["variable_descriptions"], dict):
                    return False
                
                described_vars = set(data["variable_descriptions"].keys())
                expected_vars = set(self._variables)
                if described_vars != expected_vars: return False
                
                polished_results = data
                return True
            
            except Exception:
                return False
        
        # utilize get_answer's check_and_accept feature
        _ = get_answer(
            prompt = prompt,
            model = self._auto_polisher,
            system_prompt = system_prompt,
            temperature = 0.1,
            trial_num = 10,
            check_and_accept = check_and_accept,
        )
        
        if self._input_description is None: 
            self._input_description = polished_results["input_description"]
        if self._variable_descriptions is None:
            self._variable_descriptions = polished_results["variable_descriptions"]
        if self._output_description is None:
            self._output_description = polished_results["output_description"]


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
            report_lines.append(f"{var_name}: Range [{x_min[i]:.4g}, {x_max[i]:.4g}]")
        
        report_lines.append("")
        
        y_min = y_data.min()
        y_max = y_data.max()
        report_lines.append(f"Output range: [{y_min:.4g}, {y_max:.4g}]")

        report_lines.append("")
        report_lines.append(f"Number of samples: {n_samples}")
        report_lines.append(f"Output mean: {np.mean(y_data):.4g}")
        report_lines.append(f"Output standard deviation: {np.std(y_data):.4g}")
        
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
        
        report_lines.append(f"Fit Error: MSE={mse:.3e}, MAE={mae:.3e}, Max Error={max_error:.3e}")
        
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
            if self._variable_descriptions is None:
                self._variable_descriptions = {}
            
            if list(self._variable_descriptions.keys()) != self._variables:
                for var in self._variables:
                    if var not in self._variable_descriptions:
                        self._variable_descriptions[var] = "Unknown physical quantity"
            if self._output_description is None:
                self._output_description = "Unknown physical quantity"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are physical quantities, "
                f"with respective units of {', '.join(self._variable_units)}, "
                f"and their meanings are as follows: {', '.join([var +':'+self._variable_descriptions[var] for var in self._variables])}.\n"
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
            
            if self._variable_descriptions is None:
                self._variable_descriptions = {}

            if self._variable_descriptions.keys() != self._variables:
                for var in self._variables:
                    if var not in self._variable_descriptions:
                        self._variable_descriptions[var] = "Unknown quantity"
                        
            if self._output_description is None:
                self._output_description = "Unknown quantity"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are physical quantities, "
                f"with respective units of {', '.join(self._variable_units)}, "
                f"and their meanings are as follows: {', '.join([var +':' + self._variable_descriptions[var] for var in self._variables])}.\n"
                "To simplify the problem, all optimizable parameters, denoted as `param`, are dimensionless (unit of 1). The number of these empirical parameters should be minimized.\n"
                f"Your task is to construct an expression using these physical quantities and empirical parameters to describe a physical quantity `{self._output_name}` with units of `{self._output_unit}`."
                f"The meaning of this quantity is: {self._output_description}. It is crucial to ensure the dimensional correctness of the expression."
                f"To achieve this, you may need to construct dimensionless quantities for complex function operations, and subsequently match the dimensions to the target output `{self._output_name}`."
            )
            
            if self._input_description:
                variables_info = self._input_description + "\n" + variables_info
                
        else:
            assert self._output_name is not None
            
            if self._variable_descriptions is None:
                self._variable_descriptions = {}

            if self._variable_descriptions.keys() != self._variables:
                for var in self._variables:
                    if var not in self._variable_descriptions:
                        self._variable_descriptions[var] = "[Undefined Meaning]"
                        
            if self._output_description is None:
                self._output_description = "[Undefined Meaning]"

            variables_info = (
                f"Additionally, {', '.join(self._variables)} are the input quantities, "
                f"and their meanings are as follows: {', '.join([var +':' + self._variable_descriptions[var] for var in self._variables])}.\n"
                "To simplify the problem, all optimizable empirical parameters, denoted as `param`, should be as few as possible and preferably dimensionless.\n"
                f"Your task is to construct an expression using these quantities and empirical parameters to describe a target quantity: `{self._output_name}`."
                f"The meaning of this target quantity is: {self._output_description}."
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
            trial_num = self._optimization_trial_num,
            method = self._optimization_method,
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
            
            
    def _update_pareto_frontier(
        self,
        numeric_ansatz: str,
        metric_value: float,
        info_dict: Dict[str, Any],
    )-> None:
        
        with self._pareto_frontier_lock:
            
            complexity = get_pareto_complexity(
                numeric_ansatz = numeric_ansatz,
            )
            assert isinstance(complexity, int)
                
            metric_type = "reduced chi squared" \
                if (self._error is not None) else "mean square error"
            is_dominated = False
            
            # maintain pareto frontier
            dominated_keys = []
            for existing_complexity, existing_info_dict in self._pareto_frontier.items():
                existing_metric_value = existing_info_dict[f"{metric_type}"]
                if existing_complexity <= complexity and existing_metric_value <= metric_value:
                    is_dominated = True; break
                if complexity <= existing_complexity and metric_value <= existing_metric_value:
                    dominated_keys.append(existing_complexity)
            if is_dominated: return
            for key in dominated_keys: del self._pareto_frontier[key]
            self._pareto_frontier[complexity] = info_dict

            # sync pareto data 
            with open(
                file = self._pareto_data_path, 
                mode = "w", 
                encoding = "UTF-8",
            ) as file_pointer:
                json.dump(
                    self._pareto_frontier, 
                    file_pointer, 
                    indent = 4, 
                    ensure_ascii = False,
                )

            # sync pareto report
            with open(
                file = self._pareto_report_path, 
                mode = "w", 
                encoding="UTF-8",
            ) as file_pointer:
                
                file_pointer.write("="*50 + "\n")
                file_pointer.write("           Pareto Frontier Report\n")
                file_pointer.write("="*50 + "\n\n")

                sorted_frontier = sorted(self._pareto_frontier.items())

                for complexity, info in sorted_frontier:
                    metric_val = info.get(metric_type, "N/A")
                    if isinstance(metric_val, (float, int)):
                        metric_str = f"{metric_val:.7g}"
                    else:
                        metric_str = "N/A"
                    file_pointer.write(f"Complexity: {complexity}\n")
                    file_pointer.write(f"{metric_type.title()}: {metric_str}\n")
                    file_pointer.write(f"Formula: {info.get('ansatz', 'N/A')}\n")
                    file_pointer.write(f"Timestamp: {info.get('created_at', 'N/A')}\n")
                    file_pointer.write(f"Best Parameters:{info.get('best_parameters', 'N/A')}\n\n")
                    file_pointer.write("-" * 50 + "\n\n")
