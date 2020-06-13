import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('kc_house_data.csv')

data.info()
data.head()

correlation = data.corr(method = 'pearson')
plt.figure(figsie(21,21))
sb.heatmap(correlation, xticklabaels=correlation.columns, yticklabels=correlation.columns, annot=True, linewidth=1)

training_data, test_data = train_test_split(data, train_size=0.8, random_state=0)
regress = LinearRegression()

x_train = np.array()
