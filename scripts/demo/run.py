import numpy as np
from os.path import sep as seperator
from IdeaSearch import IdeaSearcher
from IdeaSearch_fit import IdeaSearchFitter


def main():
    
    # 创建 IdeaSearcher 和 IdeaSearchFitter 的实例
    ideasearcher = IdeaSearcher()
    ideasearch_fitter = IdeaSearchFitter(
        # 参数 data 和 data_path 选一个填入
        # 若选 data_path ，文件应为 .npz 格式，且 load 后同样含有键 `x`，`y` 和 `error` （可选）
        data = {
            "x": (x := np.random.uniform(1, 10, (1000, 1))),
            "y": (5 * np.exp(-(x[:,0] - 5)**2 / (2 * 3**2)))\
                # + (noise := np.random.normal(0, 0.1, size = 1000)),
            # "error": np.abs(noise) * 1.5,
        }
    )
    
    # 首先设置数据库路径，ideasearcher 对文件系统的一切操作都会在数据库中进行
    ideasearcher.set_database_path(f"scripts{seperator}demo{seperator}database")
    
    # 绑定 ideasearch_fitter
    # prompt、评估函数和 initial_ideas 都会外包给 ideasearch_fitter
    ideasearcher.bind_helper(ideasearch_fitter)
    
    # 在 ideasearcher 中设置其它必要和可选参数
    ideasearcher.set_program_name("IdeaSearch-fit demo")
    
    sample_temperature = 40.0
    ideasearcher.set_samplers_num(3)
    ideasearcher.set_sample_temperature(sample_temperature)
    
    ideasearcher.set_evaluators_num(3)
    ideasearcher.set_examples_num(5)
    ideasearcher.set_generate_num(2)

    ideasearcher.set_api_keys_path("api_keys.json")
    ideasearcher.set_models(["Gemini_2.5_Flash"])
    
    # 开始 IdeaSearch
    island_num = 4
    cycle_num = 20
    unit_interaction_num = 20
    
    for _ in range(island_num):
        ideasearcher.add_island()
        
    for cycle in range(cycle_num):
        
        if cycle != 0:
            ideasearcher.repopulate_islands()
    
        ideasearcher.run(unit_interaction_num)
        
        # 通过 get_bset_fit 动作实时查看最优拟合函数
        print(ideasearch_fitter.get_best_fit())
    
    
if __name__ == "__main__":
    
    main()
    
    