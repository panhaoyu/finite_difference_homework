# 这个模块用于计算第二次作业第1题

import numpy as np
from core.parabolic_1d import Parabolic1D

calculator = Parabolic1D(1, 0.1, 1)
result = calculator.forward_difference(data=np.arange(2, 9), left=1, right=9)
print()
