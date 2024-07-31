# Problem Statement: predicting Petal Length using Sepal Length


#type:ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

#Data Loading
data = pd.read_csv('Iris.csv')

#2. Data Cleaning
data = data.dropna()
data = data.drop_duplicates()

# 3.Taking out data
sepal_length = data['SepalLengthCm']
petal_length = data['PetalLengthCm']

X = sepal_length.values.reshape(-1,1) #-1 batata hai k tamam values lo aor 1 batata hai k ek column main dalo
Y = petal_length.values.reshape(-1,1)

#4. Splitting Data
x_train, x_test, y_train, y_test = train_test_split (X,Y, test_size=0.3, random_state=34)

# 5. Plotting to know that which model should be applied
plt.scatter(x_train, y_train, color = 'blue', label = 'Training Data')
plt.scatter(x_test, y_test, color= 'red', label = 'Testing Data')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Scatter Plot of Sepal Length vs Petal Length')
plt.legend()
plt.show()
# Higher Corellation the Linear regression if not very related then polynomial regression
correlation_coeffieinct = pearsonr (sepal_length, petal_length)
print ('Correlation Coefficient:',correlation_coeffieinct)

# 6 Model Training
model = LinearRegression()
model.fit(x_train,y_train)

# 7 Model Prediction Y from X
y_training_prediction = model.predict(x_train)
y_testing_prediction = model.predict(y_test)

#8 Find Overfitting/underfitting

mse_trained = mean_squared_error(y_train,y_training_prediction)
mse_test = mean_squared_error(y_test, y_testing_prediction) #Y is our outout here

# Performance of Linear Regression
r2_train = r2_score(y_train, y_training_prediction)
r2_testing = r2_score(y_test, y_testing_prediction)

print('Training Mean Squared Error:',mse_trained)
print('Test Mean Squared Error:', mse_test)
print('Training R^2 Score:',r2_train)
print('Test R^2 Score:', r2_testing)

sepal_length_new = np.array([[3.8]])
petal_length_pred = model.predict(sepal_length_new)
print('Predicted Petal Length for Sepal Length 3.8 cm:', petal_length_pred[0])








