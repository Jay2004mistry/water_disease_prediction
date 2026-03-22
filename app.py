

#flask:- create API server
#request :- get the data from frontend
#jsonify:- sne djson response 
from flask import Flask,request,jsonify

#cors:-allow cors request means cross platform
from flask_cors import CORS

#pickle :- help to load models save as .pkl
import pickle

#numpy:-handle array
import numpy as np


#   Flask(__name__) → starts backend app
app=Flask(__name__)
# CORS(app) → allows frontend (localhost:3000) to talk with Flask (to connect)
CORS(app, origins=[
    "https://jay2004mistry.github.io",
    "http://127.0.0.1:5500",
    "http://localhost:5500"
])

#load (file and scaler) once  after connect with API
with open('Linear_reg_model_for_disease_water.pkl','rb') as f:
    model=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)



#create api end point (http://localhost:5000/predict)
@app.route('/predict',methods=['POST'])

#This function runs when frontend hits /predict
def predict():

    #get the data from frontend(textbox)
    data=request.json

    #conver data into array because model understand (array 2d array)
    #.reshape(1, -1) → makes it 2D array
    features=np.array(data['features']).reshape(1,-1)

    #Converts input into same format as training (as in dataset)
    scaled = scaler.transform(features)

    #make predication and model give output like ([[12.3, 5.6, 8.9, 2.1]])
    prediction=model.predict(scaled)
    
    # clip — never allow negative predictions
    prediction = np.clip(prediction, 0, None)  # min=0, max=no limit


#send the response 
#prediction[0][1] because output will be like  ([[value1, value2, value3, value4]]) //2d array
    return jsonify({
    'Diarrheal'       : round(float(prediction[0][0]), 2),
    'Cholera'         : round(float(prediction[0][1]), 2),
    'Typhoid'         : round(float(prediction[0][2]), 2),
    'Infant_Mortality': round(float(prediction[0][3]), 2)
})

#main method
if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)


#full flow
"""React (Frontend)
    ↓ POST request
Flask API (/predict)
    ↓
request.json (get data)
    ↓
numpy (convert)
    ↓
scaler.transform()
    ↓
model.predict()
    ↓
jsonify()
    ↓
React receives result"""