import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from flask import Flask, request, render_template

dataset = pd.read_csv('dataset/train_dataset.csv', index_col=0)

x = dataset.iloc[:, : 10].values
y = dataset.iloc[:, 10].values

scaler = StandardScaler()
scaler.fit_transform(x)

model = tf.keras.models.load_model('model/project_model.h5')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
    v1 = trans_datetime.hour
    v2 = trans_datetime.day
    v3 = trans_datetime.month
    v4 = trans_datetime.year
    v5 = int(request.form.get("category"))
    v6 = float(request.form.get("card_number"))
    dob = pd.to_datetime(request.form.get("dob"))
    v7 = np.round((trans_datetime - dob) // np.timedelta64(1, 'Y'))
    v8 = float(request.form.get("trans_amount"))
    v9 = int(request.form.get("state"))
    v10 = int(request.form.get("zip"))
    x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])
    y_pred = model.predict(scaler.transform([x_test]))
    if y_pred[0][0] <= 0.5:
        result = "VALID TRANSACTION"
    else:
        result = "FRAUD TRANSACTION"
    return render_template('result.html', OUTPUT='{}'.format(result))

if __name__ == "__main__":
    app.run()
