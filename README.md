# About IdeaSearch-fit

## Quick Start

To install IdeaSearch-fit, run:

```bash
pip install IdeaSearch-fit
```

## Project Overview

`IdeaSearch-fit` is a **scientific formula discovery tool focused on symbolic regression, built upon the `IdeaSearch` framework**. It provides the `IdeaSearchFitter` class, which is designed as a highly integrated "Helper" object that seamlessly interfaces with `IdeaSearch` via the `bind_helper` method. `IdeaSearch-fit` integrates the dual capabilities of evolutionary computation and large language models to automatically discover underlying mathematical expressions from experimental or simulation data.

The core task of `IdeaSearch-fit` is to solve the symbolic regression problem: given a set of input data $(X)$ and a corresponding output $(y)$, the system performs an evolutionary search to find a mathematical formula $f(X)$ that is both accurate in describing the relationship and parsimonious in its complexity. Therefore, it is more than just a data fitting tool; its ultimate goal is to be an exploratory framework capable of uncovering the physical laws or mathematical patterns behind the data.

## Core Methodology: Combining Heuristic Search and Numerical Optimization

The core of `IdeaSearch-fit` lies in its hybrid workflow, which combines the macroscopic structural exploration of Large Language Models (LLMs) with the microscopic parameter optimization of traditional numerical methods:

1.  **Phase 1: Ansatz Search via Large Language Models**
    This phase leverages the **instruction-following and pattern-recognition capabilities of LLMs to conduct a heuristic search for candidate formulas** within a space composed of mathematical symbols, physical concepts, and natural language. Guided by user-provided data features, physical units, and optional theoretical context, the LLM generates a series of structurally sound candidate mathematical expressions (i.e., "ansatze"). This is a macroscopic, exploratory process of structure generation.

2.  **Phase 2: Parameter Fitting via Numerical Optimization**
    Once a promising ansatz is generated (e.g., `param1 * sin(param2 * x)`), the `IdeaSearchFitter` instance immediately employs **user-specified traditional numerical optimization algorithms (e.g., `L-BFGS-B` or `differential-evolution`)** to precisely solve for the optimal values of the undetermined parameters (`param1`, `param2`, etc.). This is a deterministic, microscopic process of parameter fitting, designed to accurately evaluate the fitting capability of each candidate structure.

This workflow, where the "LLM explores structures and numerical optimization solves for parameters," integrates the structural generation capabilities of modern AI with the precise computational power of classical algorithms, forming the foundation of `IdeaSearch-fit`.

## Core Input and Output

From a user's perspective, the workflow of `IdeaSearch-fit` is designed to be exceptionally direct:

**Core Input:**

- **`X`**: Input data (independent variables, multi-dimensional support).
- **`y`**: Target data (dependent variable, one-dimensional).
- **`error`** (Optional): The measurement error for each data point in `y`. Providing errors is fundamental for rigorous scientific data fitting, as it enables the use of more robust metrics like the reduced chi-squared statistic.

**Core Output:**

- **A Pareto Frontier**. The system's output is not a single "best" formula but a set of solutions. Each formula in this set represents the optimal accuracy for its respective level of complexity, allowing the user to make a trade-off between the model's **simplicity** and **precision** that best suits their domain requirements.

## Key Features

- **Evolutionary Symbolic Regression**: Leverages the multi-island parallel evolution framework of `IdeaSearch` as its core engine to efficiently search the vast space of mathematical expressions for optimal fitting formulas.

- **Pareto-Optimal Frontier**: `IdeaSearch-fit` does not seek a single best fit but simultaneously optimizes two competing objectives: **accuracy (Metric) and simplicity (Complexity)**. It generates a **Pareto Frontier** report, providing a suite of optimal solutions at various complexity-accuracy trade-offs for in-depth analysis.

- **Rigorous Dimensional Analysis**: Includes a built-in unit validation module (`unit_validator`) that performs strict physical unit checks during the formula evolution process. This feature is crucial for scientific and engineering domains, ensuring that the generated formulas are valid not only numerically but also physically.

- **Dual-Mode Formula Generation**:

  - **Precise Generation**: Directly generates and evolves mathematical expressions that strictly adhere to a specific computational syntax.
  - **Fuzzy Generation**: First, an LLM acts as a scientist to propose a **natural language theory** about the underlying patterns in the data. Then, another LLM agent translates this theory into a strict mathematical expression. This innovative two-stage approach allows for the infusion of richer prior knowledge and creative insights.

- **Automated Semantic Polishing**: Users need only provide basic variable names and units, and the system can invoke an LLM to automatically infer and complete descriptions of the physical meaning of all inputs and outputs. This information is then used to construct more semantically rich prompts, guiding `IdeaSearch` toward a more intelligent exploration.

- **Seamless Integration and Automation**: As a highly integrated module for `IdeaSearch`, the `IdeaSearchFitter` class only needs to be initialized once. It then automatically configures all the necessary core components for `IdeaSearch`, including the evaluation function (`evaluate_func`), prompts (`prologue_section`, `epilogue_section`), and mutation/crossover operators (`mutation_func`, `crossover_func`).

## Core API

The primary interface for the `IdeaSearch-fit` package is the `IdeaSearchFitter` class, which consists of its constructor and several result-retrieval methods.

- `IdeaSearchFitter(__init__)`: ⭐️ **Important**

  - **Function**: Initializes an `IdeaSearchFitter` instance. This is the unified entry point for all configurations, including data loading, problem definition, expression construction, and search strategy settings.
  - **Importance**: The first step in using `IdeaSearch-fit`, defining the entire framework for the symbolic regression task.

- `get_best_fit()`:

  - **Function**: Returns the numerical expression string, fitted with its optimal parameters, that achieved the lowest metric value (e.g., mean squared error) among all evaluated formulas.
  - **Importance**: Retrieves the single best fitting result based on accuracy alone.

- `get_pareto_frontier()`:

  - **Function**: Returns a dictionary containing all formulas on the current Pareto frontier, along with their detailed information (complexity, metric value, fitted parameters, etc.).
  - **Importance**: Retrieves a set of optimal solutions that balance accuracy and complexity.

## Configuration Parameters

All configurations for an `IdeaSearchFitter` instance are set in its `__init__` constructor. The parameters are logically grouped by function to facilitate understanding and setup.

### 1\. Task Input & Output

- **`data`**: `Optional[Dict[str, ndarray]]`
  - Data passed directly as a dictionary in memory. Must contain keys `"x"` (input, 2D array) and `"y"` (output, 1D array). Can optionally include `"error"` (errors for y, 1D array).
- **`data_path`**: `Optional[str]`
  - Path to a local `.npz` file from which to load data. Use either `data` or `data_path`. The file should contain arrays with the same keys.
- **`result_path`**: `str`
  - Path to an **existing directory** where fitting results, such as the Pareto frontier report (`pareto_report.txt`) and data (`pareto_data.json`), will be stored.

### 2\. Problem Definition & Units

- **`variable_names`**: `Optional[List[str]]`
  - A list of names for the input variables (e.g., `["mass", "velocity"]`). Defaults to `["x1", "x2", ...]` if not provided.
- **`output_name`**: `Optional[str]`
  - The name of the output variable (e.g., `"energy"`).
- **`perform_unit_validation`**: `bool` (Default: `False`)
  - **[Master Switch]** Whether to enable strict physical dimensional analysis.
- **`variable_units`**: `Optional[List[str]]`
  - A list of units corresponding to `variable_names` (e.g., `["kg", "m s^-1"]`). Required when `perform_unit_validation` is `True`.
- **`output_unit`**: `Optional[str]`
  - The unit of the output variable (e.g., `"J"` or `"kg m^2 s^-2"`). Required when `perform_unit_validation` is `True`.
- **`auto_polish`**: `bool` (Default: `True`)
  - Whether to automatically invoke an LLM to infer and generate descriptions for the physical meaning of variables to enrich prompts.
- **`auto_polisher`**: `Optional[str]`
  - The name of the LLM model to use for the `auto_polish` task. If `None`, an available model is selected automatically.
- **`input_description`**, **`variable_descriptions`**, **`output_description`**: `Optional[str]`
  - Manually specify text descriptions for the system, input variables, and output variable. If provided, the `auto_polish` step will be skipped.

### 3\. Expression Building Blocks

- **`functions`**: `List[str]` (Default: Includes common functions like `sin`, `cos`, `log`, `exp`)
  - A list of mathematical functions allowed in expressions. All functions must be supported by the `numexpr` library.
- **`constant_whitelist`**: `List[str]` (Default: `[]`)
  - A list of named constants (as strings) allowed in expressions.
- **`constant_map`**: `Dict[str, float]` (Default: `{"pi": np.pi}`)
  - A dictionary mapping named constants to their float values.

### 4\. Search & Evolution Strategy

- **LLM Generation Strategy**
  - **`generate_fuzzy`**: `bool` (Default: `True`)
    - **[Master Switch]** Enables "fuzzy generation" mode. If `False`, the LLM will generate strict mathematical expressions directly.
  - **`fuzzy_translator`**: `Optional[str]`
    - In fuzzy mode, the name of the LLM model used to translate natural language theories into mathematical expressions.
- **Numerical Optimization Strategy**
  - **`optimization_method`**: `Literal["L-BFGS-B", "differential-evolution"]` (Default: `"L-BFGS-B"`)
    - Specifies the numerical optimization algorithm for fitting ansatz parameters. `L-BFGS-B` is an efficient local optimizer suitable for smoother objective functions. `differential-evolution` is a global optimizer that may be more robust for complex problems with multiple local minima.
  - **`optimization_trial_num`**: `int` (Default: `100`)
    - The number of trials for the numerical optimizer. For `differential-evolution`, this corresponds to the population size. For non-convex or complex problems, increasing this value can improve the probability of finding the global optimum at the cost of increased computation time.
- **Evolutionary Operators (Still work in progress)**
  - **`enable_mutation`**: `bool` (Default: `False`)
    - Enables the built-in expression **mutation** functionality.
  - **`enable_crossover`**: `bool` (Default: `False`)
    - Enables the built-in expression **crossover** functionality.

### 5\. Score Mapping

- **`baseline_metric_value`**: `Optional[float]`
  - Defines a "baseline" metric value (e.g., MSE) that will be mapped to an `IdeaSearch` score of `20.0`. If `None`, the result of a naive linear fit is used as the baseline.
- **`good_metric_value`**: `Optional[float]`
  - Defines a "good" metric value that will be mapped to an `IdeaSearch` score of `80.0`.
- **`metric_mapping`**: `Literal["linear", "logarithm"]` (Default: `"linear"`)
  - The mapping function from metric value to score. Logarithmic mapping is more suitable when metric values can span several orders of magnitude.

### 6\. Miscellaneous

- **`seed`**: `Optional[int]`
  - A random seed to ensure the reproducibility of results.

## Workflow

The following is a complete workflow example (from [https://github.com/IdeaSearch/IdeaSearch-fit-test](https://github.com/IdeaSearch/IdeaSearch-fit-test)) that demonstrates how to solve a symbolic regression problem using `IdeaSearch-fit` and `IdeaSearch`.

```python
# pip install IdeaSearch-fit
import numpy as np
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def build_data():
    """
    Builds simulated data for fitting.
    True Formula: x = 1.2*A + A*exp(-0.7*gamma*t) - A*exp(-0.5*gamma*t)*cos(3*omega*t)
    """
    n_samples = 1000
    rng = np.random.default_rng(seed=42)

    # Define the value ranges for the independent variables
    A_range = (1.0, 10.0)
    gamma_range = (0.1, 1.0)
    omega_range = (1.0, 5.0)
    t_range = (0.0, 10.0)

    # Sample randomly within the defined ranges
    A_samples = rng.uniform(A_range[0], A_range[1], n_samples)
    gamma_samples = rng.uniform(gamma_range[0], gamma_range[1], n_samples)
    omega_samples = rng.uniform(omega_range[0], omega_range[1], n_samples)
    t_samples = rng.uniform(t_range[0], t_range[1], n_samples)

    # Stack the independent variables into the format required by the Fitter (n_samples, n_features)
    x_data = np.stack([A_samples, gamma_samples, omega_samples, t_samples], axis=1)

    # Calculate the y values based on the true formula and add noise
    y_true = (1.2 * A_samples +
              A_samples * np.exp(-0.7 * gamma_samples * t_samples) -
              A_samples * np.exp(-0.5 * gamma_samples * t_samples) * np.cos(3 * omega_samples * t_samples))
    error_data = 0.01 + 0.02 * np.abs(y_true)
    y_data = y_true + rng.normal(0, error_data)

    # Note: This example does not pass error_data to the Fitter, so MSE will be used as the metric.
    return {
        "x": x_data,
        "y": y_data,
    }


def main():
    # 1. Prepare the data
    data = build_data()

    # 2. Initialize IdeaSearchFitter
    fitter = IdeaSearchFitter(
        result_path = "fit_results", # Ensure this directory exists
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

    # 3. Initialize IdeaSearcher
    ideasearcher = IdeaSearcher()

    # 4. Basic Configuration
    ideasearcher.set_program_name("IdeaSearch Fitter Test")
    ideasearcher.set_database_path("database") # Ensure this directory exists
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models(["gemini-2.5-pro"]) # Use your desired model

    ideasearcher.set_record_prompt_in_diary(True) # Optional: Record prompts in the diary log

    # 5. Bind the Fitter! (The most crucial step)
    ideasearcher.bind_helper(fitter)

    # 6. Define and execute the evolutionary loop
    island_num = 3
    cycle_num = 3
    unit_interaction_num = 20

    for _ in range(island_num):
        ideasearcher.add_island()

    for cycle in range(cycle_num):
        print(f"---[ Cycle {cycle + 1}/{cycle_num} ]---")
        if cycle != 0: ideasearcher.repopulate_islands()
        ideasearcher.run(unit_interaction_num)
        print("Done.")

    # 7. Retrieve and display the results
    print("\n---[ Search Complete ]---")
    print(f"Best Fit Formula: {fitter.get_best_fit()}")

    print("\nOptimal Solutions on the Pareto Frontier:")
    pareto_frontier = fitter.get_pareto_frontier()
    if not pareto_frontier:
        print("  - No solutions found on the Pareto frontier.")
    else:
        for complexity, info in sorted(pareto_frontier.items()):
            metric_key = "reduced chi squared" if "reduced chi squared" in info else "mean square error"
            metric_value = info.get(metric_key, float('nan'))
            print(f"  - Complexity: {complexity}, Error: {metric_value:.4g}, Formula: {info.get('ansatz', 'N/A')}")


if __name__ == "__main__":
    # Before running, please ensure:
    # 1. The './fit_results' and './database' directories exist.
    # 2. The 'api_keys.json' file is correctly configured.
    main()
```
