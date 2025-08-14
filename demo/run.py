import numpy as np
from os.path import sep as seperator
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def main():
    
    ideasearcher = IdeaSearcher()
    
    ideasearch_fitter = IdeaSearchFitter(
        # 参数 data 和 data_path 选一个填入
        # 若选 data_path ，文件应为 .npz 格式，且 load 后同样含有键 `x`，`y` 和 `error` （可选）
        data = {
            "x": (x := np.random.uniform(0, 2 * np.pi, (1000, 2))),
            "y": (2 * np.sin(x[:, 0]) + 3 * np.cos(x[:, 1]))\
                + (noise := np.random.normal(0, 0.1, size = 1000)),
            "error": np.abs(noise) * 1.5,
        }
    )
    
    # prompt、评估函数和 initial_ideas 都外包给 ideasearch_fitter
    ideasearcher.bind_helper(ideasearch_fitter)
    
    # 在 ideasearcher 中设置其它必要和可选参数
    ideasearcher.set_program_name("TransferFunction")
    ideasearcher.set_database_path(f"demo{seperator}database")
    
    sample_temperature = 40.0
    ideasearcher.set_samplers_num(3)
    ideasearcher.set_sample_temperature(sample_temperature)
    ideasearcher.set_evaluators_num(3)
    ideasearcher.set_examples_num(5)
    ideasearcher.set_generate_num(2)
    
    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models([
        "Gemini_2.5_Flash",
        "Gemini_2.5_Flash",
        "Gemini_2.5_Flash",
        "Gemini_2.5_Flash",
        "Gemini_2.5_Flash",
    ])
    ideasearcher.set_model_temperatures([
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
    ])
    ideasearcher.set_generation_bonus(3.0)
    
    # 手动设置一下初始 ideas
    ideasearcher.add_initial_ideas(
        ideasearch_fitter.initial_ideas
    )
    
    island_num = 4
    cycle_num = 20
    unit_interaction_num = 20
    
    for _ in range(island_num):
        ideasearcher.add_island()
        
    for cycle in range(cycle_num):
        
        if cycle != 0:
            ideasearcher.repopulate_islands()
    
        ideasearcher.run(unit_interaction_num)
    
    
if __name__ == "__main__":
    
    main()
    
    