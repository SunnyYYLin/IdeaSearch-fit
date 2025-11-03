# pip install IdeaSearch-fit
import numpy as np
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def mock_data():
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
    data = mock_data()

    # 2. Initialize IdeaSearchFitter
    fitter = IdeaSearchFitter(
        result_path = "results", # Ensure this directory exists
        data = data,
        variable_names = ["A", "gamma", "omega", "t"],
        variable_units = ["m", "s^-1", "s^-1", "s"],
        output_name = "x",
        output_unit = "m",
        constant_whitelist = ["1", "2", "pi"],
        constant_map = {"1": 1, "2": 2, "pi": np.pi},
        input_description = "这是一个模拟的物理过程，A为幅值，gamma和omega为速率参数，t为时间，x为观测量。",
        auto_polish = False,
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
    ideasearcher.set_models(["gpt-5-mini"]) # Use your desired model

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