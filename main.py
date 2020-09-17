import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('kc_house_data.csv')

data.info()
data.head()

correlation = data.corr(method='pearson')
plt.figure(figsize=(21, 21))
sb.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True, linewidth=1)

training_data, testing_data = train_test_split(data, train_size=0.8, random_state=0)
regress = LinearRegression()

x_train = np.array(testing_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)
y_train = np.array(testing_data['price'], dtype=pd.Series)
regress.fit(x_train, y_train)

x_test = np.array(testing_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)
y_test = np.array(testing_data['price'], dtype=pd.Series)

prediction = regress.predict(x_test)

plt.figure(figsize=(10,8))
plt.plot(x_test, prediction, color='green', label="Regression Line")
plt.scatter(x_test, y_test, color='black', label="Testing Data", alpha=0.5, edgecolors='darkgray')
plt.xlabel("sqft_living", fontsize=20)
plt.ylabel("price", fontsize=20)