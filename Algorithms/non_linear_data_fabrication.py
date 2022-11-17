import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg as kr 
import plotly.express as px
import plotly.graph_objects as go



# Createing non linear data
x = np.linspace(-10, 10, 100) + np.random.normal(0, 1, 100)
y = x**2 + np.random.randn(len(x)) * 10

# Plotting the data
plt.scatter(x, y)
plt.show()

# # fiting line useing extermal modules 
# fig = px.scatter(x=x, y=y)

# fig.add_trace(go.Scatter(x= x, y= y, name='Statsmodels fit',  mode='lines'))
# fig.show()

# saveing the data
df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('non_linear_data.csv', index=False)
