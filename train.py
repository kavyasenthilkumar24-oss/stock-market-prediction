import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("d1.csv")

X = df[['Open','High','Low','Close']]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model Saved")