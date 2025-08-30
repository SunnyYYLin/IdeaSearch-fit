from collections import namedtuple


__all__ = [
    "validate_unit_testcases",
]


ValidateUnitTestcase = namedtuple(
    "ValidateUnitTestcase",
    [
        "description",
        "expression",
        "expression_unit",
        "variable_names",
        "variable_units",
        "expected_correctness",
    ]
)


DIMLESS =    [0, 0, 0, 0, 0]
LENGTH =     [1, 0, 0, 0, 0]
TIME =       [0, 1, 0, 0, 0]
MASS =       [0, 0, 1, 0, 0]
VELOCITY =   [1, -1, 0, 0, 0]
ACCEL =      [1, -2, 0, 0, 0]
FORCE =      [1, -2, 1, 0, 0]
ENERGY =     [2, -2, 1, 0, 0]
PRESSURE =   [-1, -2, 1, 0, 0]
FREQUENCY =  [0, -1, 0, 0, 0]
VOLUME =     [3, 0, 0, 0, 0]
TEMP =       [0, 0, 0, 1, 0]
GAS_CONST =  [2, -2, 1, -1, 0]
GRAV_CONST = [3, -2, -1, 0, 0]


validate_unit_testcases = [
    
    ValidateUnitTestcase(
        description = "正确表达式 (F = m * a)",
        expression = "m * x / y",
        expression_unit = FORCE,
        variable_names = ['m', 'x', 'y'],
        variable_units = [MASS, VELOCITY, TIME],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "错误表达式 (力 = 动量)",
        expression = "m * x",
        expression_unit = FORCE,
        variable_names = ['m', 'x'],
        variable_units = [MASS, VELOCITY],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "量纲不兼容的加法 (质量 + 速度)",
        expression = "m + x",
        expression_unit = MASS,
        variable_names = ['m', 'x'],
        variable_units = [MASS, VELOCITY],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "正确表达式 (y / x**2)",
        expression = "y / (x**2)",
        expression_unit = MASS,
        variable_names = ['x', 'y'],
        variable_units = [[1, -1, 0, 0, 0], [2, -2, 1, 0, 0]],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "错误量纲 (y / x**2)",
        expression = "y / (x**2)",
        expression_unit = [1, 1, 1, 1, 1],
        variable_names = ['x', 'y'],
        variable_units = [[1, -1, 0, 0, 0], [2, -2, 1, 0, 0]],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "复杂表达式 (角动量)",
        expression = "sqrt(r1**2+r2**2+2*r1*r2*cos(omega*(t1-t2)+phi))*sin(xi)*m*v",
        expression_unit = [2, -1, 1, 0, 0], # 角动量
        variable_names = ['r1', 'r2', 'omega', 't1', 't2', 'phi', 'xi', 'm', 'v'],
        variable_units = [LENGTH, LENGTH, FREQUENCY, TIME, TIME, DIMLESS, DIMLESS, MASS, VELOCITY],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "带根号的减法 (sqrt(L1-L2))",
        expression = "sqrt(r1 - 10 * r2)",
        expression_unit = [0.5, 0, 0, 0, 0],
        variable_names = ['r1', 'r2'],
        variable_units = [LENGTH, LENGTH],
        expected_correctness = True,
    ),

    ValidateUnitTestcase(
        description = "无量纲参数 (param)",
        expression = "param1 * x + sin(param2) - param3 / (y + param4) + cos(z)",
        expression_unit = DIMLESS,
        variable_names = ["x", "y", "z"],
        variable_units = [DIMLESS, DIMLESS, DIMLESS],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "动能 (E = 0.5 * m * v^2)",
        expression = "0.5 * m * v**2",
        expression_unit = ENERGY,
        variable_names = ['m', 'v'],
        variable_units = [MASS, VELOCITY],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "不兼容加法 (能量 + 速度)",
        expression = "m * v**2 + v",
        expression_unit = ENERGY,
        variable_names = ['m', 'v'],
        variable_units = [MASS, VELOCITY],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "三角函数输入有量纲 (sin(L))",
        expression = "sin(L)",
        expression_unit = DIMLESS,
        variable_names = ['L'],
        variable_units = [LENGTH],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "三角函数输入无量纲 (L*sin(w*t))",
        expression = "L * sin(w * t)",
        expression_unit = LENGTH,
        variable_names = ['L', 'w', 't'],
        variable_units = [LENGTH, FREQUENCY, TIME],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "对数函数输入有量纲 (log(m))",
        expression = "log(m)",
        expression_unit = DIMLESS,
        variable_names = ['m'],
        variable_units = [MASS],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "对数函数输入无量纲 (log(P/P_ref))",
        expression = "log(P / P_ref)",
        expression_unit = DIMLESS,
        variable_names = ['P', 'P_ref'],
        variable_units = [PRESSURE, PRESSURE],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "指数函数输入有量纲 (exp(t))",
        expression = "exp(t)",
        expression_unit = DIMLESS,
        variable_names = ['t'],
        variable_units = [TIME],
        expected_correctness = False,
    ),
    
    ValidateUnitTestcase(
        description = "指数函数输入无量纲 (A*exp(-k*t))",
        expression = "A * exp(-k * t)",
        expression_unit = LENGTH,
        variable_names = ['A', 'k', 't'],
        variable_units = [LENGTH, FREQUENCY, TIME],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "理想气体定律 (P = nRT/V)",
        expression = "n * R * T_var / V_var",
        expression_unit = PRESSURE,
        variable_names = ['n', 'R', 'T_var', 'V_var'],
        variable_units = [DIMLESS, GAS_CONST, TEMP, VOLUME],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "万有引力 (F = Gm1m2/r^2)",
        expression = "G * m1 * m2 / r**2",
        expression_unit = FORCE,
        variable_names = ['G', 'm1', 'm2', 'r'],
        variable_units = [GRAV_CONST, MASS, MASS, LENGTH],
        expected_correctness = True,
    ),
    
    ValidateUnitTestcase(
        description = "无量纲表达式，但期望有量纲",
        expression = "x / y",
        expression_unit = LENGTH,
        variable_names = ['x', 'y'],
        variable_units = [LENGTH, LENGTH],
        expected_correctness = False,
    ),
]
