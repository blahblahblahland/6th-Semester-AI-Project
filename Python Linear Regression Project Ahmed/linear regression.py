#Group members:
#Ahmed Shakil (FA21-BCS-054)
#Abdul Wassay Tahir (FA21-BCS-045)
#Hassan Jahangir Abbasi (FA21-BCS-089)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Getting the data from data.csv
data = pd.read_csv('C:/Users/PMLS/Desktop/Python Linear Regression Project Ahmed/data.csv')
#splitting the first 5 columns as x, 6th as y
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
#Training the regression model and getting the parameters
model = LinearRegression()
model.fit(X, Y)
intercept = model.intercept_
coefficients = model.coef_
#Predicting the values for the trained model
y_prediction = model.predict(X)
#Printing the values
print('Coefficients: ',coefficients,'\n Intercepts: ',intercept)
#Visualizing the predicted vs actual values using pyplot graph
plt.figure(figsize=(10, 6))
plt.scatter(Y, y_prediction, color='blue')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linewidth=2)  # Line for perfect prediction
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()
