import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100)
y = 3.5 * x + 2 + np.random.normal(0, 40, 100)

df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('data.csv', index=False)

plt.scatter(x, y)
plt.show()