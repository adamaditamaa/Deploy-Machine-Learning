from flask import flask
from flask import request
from flask import jsonfy
import pandas as pd
from modules.insurance_model import InsuranceModel

app = flask(__name__)
@app.route('/')
def home():
    return 'Welcome to API ML'

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.dataframe(data)
    result_predict = InsuranceModel().runModel(df,typed='single')
    return jsonfy({
        "status":"predicted",
        "predicted_result":result_predict
    })    

if __name__ == '__main__':
    app.run(port=9000)

