from utils import *


__all__ = [
    "get_pareto_complexity",
]


def get_pareto_complexity(
    numeric_ansatz: str,
)-> float:
    
    """
    aligned with PySR
    """
    
    try:
        tree = ast.parse(numeric_ansatz, mode='eval')
        return _count_complexity(tree.body)
    except Exception:
        return float("inf")


def _count_complexity(
    node,
):

    if isinstance(node, ast.BinOp):
        left = _count_complexity(node.left)
        right = _count_complexity(node.right)
        return 1 + left + right
    elif isinstance(node, ast.UnaryOp):
        return 1 + _count_complexity(node.operand)
    elif isinstance(node, ast.Call):
        return 1 + sum(_count_complexity(arg) for arg in node.args)
    elif isinstance(node, (ast.Name, ast.Constant, ast.Num)):
        return 1
    else:
        return 1