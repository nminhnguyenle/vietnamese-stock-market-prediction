import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("./data/price_train_final.csv")
df.dropna(inplace = True)

features = ["open", "high", "low", "volume"]
X = df[features]
y = df["close"]
rows_per_stock = 205
num_stocks = (len(df) - 1) // rows_per_stock
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
dt_model = BayesianRidge()
unique_stocks = df['symbol'].unique()
unique_stocks_list = unique_stocks.tolist()
print(unique_stocks_list)
num_stocks = len(unique_stocks_list) 
rows_per_stock = 205  
data = {}
for i in range(num_stocks):
    dt_model = BayesianRidge()
    start_row = i * rows_per_stock
    end_row = start_row + rows_per_stock - 1
    stock_data = df.iloc[start_row:end_row]
    stock = unique_stocks_list[i]
    X = stock_data[features]
    y = stock_data["close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, shuffle=False)
    
    data[f'X_train_{stock}'] = X_train  
    data[f'X_test_{stock}'] = X_test
    data[f'y_train_{stock}'] = y_train
    data[f'y_test_{stock}'] = y_test 

    dt_model.fit(data[f'X_train_{stock}'], data[f'y_train_{stock}'])
    data[f'y_preds_{stock}_array'] = dt_model.predict(data[f'X_test_{stock}']).tolist()
    y_preds = data[f'y_preds_{stock}_array']

    plt.plot(list(range(len(data[f"X_test_{stock}"]))), y_test, label = f'Actual price, {stock}')
    plt.plot(list(range(len(data[f"X_test_{stock}"]))), y_preds, label = f'Predicted price, {stock}')

plt.legend(loc = 'right', prop={'size': 2})

plt.title('Actual price vs. predicted price')
plt.xlabel('ID')
plt.ylabel('Closing price')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol = 5, prop={'size': 6})

plt.tight_layout()
plt.show()
