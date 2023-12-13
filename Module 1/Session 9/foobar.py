import numpy as np
import pandas as pd
foo = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
bar = foo.T**2

df_foo = pd.DataFrame({'baz': [1,2,3,4], 'bop': [11, 12, 13, 14]})
df_bar = pd.DataFrame({'BAZ': [1,2,3,4], 'bop': [11, 12, 13, 14]})
df_test = pd.DataFrame({'baz': [1,2,3,4], 'bop': [11, 12, 13, 14]})

stu_arr = np.ones(10000)
ins_arr = np.ones(10000)
ins_arr[[5000, 25, 723, 1995]] = 0
