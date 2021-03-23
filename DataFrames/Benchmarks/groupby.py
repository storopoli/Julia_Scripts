import pandas as pd
import numpy as np

n = 10000

df = pd.DataFrame({'x': np.random.choice(['A', 'B', 'C', 'D'], n, replace = True),
                   'y': np.random.randn(n),
                   'z': np.random.rand(n)})

# 1.73ms 949µs (M1)
%%timeit
df.groupby('x').agg({'y': 'mean', 'z': 'median'})
