from unittest import TestCase
from unittest import main as unittest_main
from .testcases import validate_unit_testcases
from .unit_validator import DIMENSION_NAMES
from .unit_validator import validate_unit


"""
注：此脚本引用的 unit_validator 需手动与打包进 fitter 里的 unit_validator 对齐
"""


class TestValidateUnit(TestCase):
    
    def test_validate_unit(self):
        
        passed_count = 0
        failed_cases = []
        
        for i, case in enumerate(validate_unit_testcases):
            
            with self.subTest(
                i = i, 
                description = case.description,
                expr = case.expression
            ):
                
                result, message = validate_unit(
                    expression = case.expression,
                    expression_unit = case.expression_unit,
                    variable_names = case.variable_names,
                    variable_units = case.variable_units,
                )

                result_correct = (result == case.expected_correctness)
                
                if result_correct:
                    passed_count += 1
                else:
                    failed_cases.append((i + 1, case.description))
                
                self.assertEqual(
                    first = result, 
                    second = case.expected_correctness, 
                    msg = (
                        f"用例 {i} 失败: {case.description}\n"
                        f"表达式: {case.expression}\n"
                        f"错误信息: {message}"
                    ),
                )
        
        print(f"\n--- 测试总结 ---")
        print(f"量纲顺序: {DIMENSION_NAMES}")
        print(f"总用例数: {len(validate_unit_testcases)}")
        print(f"通过用例数: {passed_count}")
        
        if failed_cases:
            print(f"失败用例数: {len(failed_cases)}")
            for num, desc in failed_cases:
                print(f"  - 测试 {num}: {desc}")
                
                
def main():
    
    unittest_main()


if __name__ == "__main__":
    
    main()