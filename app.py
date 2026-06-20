from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    date = request.form['date']
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])

    date_value = pd.to_datetime(date).toordinal()

    data = np.array([[date_value,
                      open_price,
                      high_price,
                      low_price]])

    prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction_text=f"Predicted Close Price: ₹{prediction:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True,port=5050)