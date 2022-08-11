# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:01:56 2022

@author: 91931
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    dataset= pd.read_excel('DATASET education.xlsx')
    X = dataset.iloc[:, 0:8].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    Tenth             = float(request.args.get('Tenth'))
    Twelfth           = float(request.args.get('Twelfth'))
    B_Tech            = float(request.args.get('B_Tech'))
    Seventh_SEM       = float(request.args.get('Seventh_SEM'))
    Sixth_SEM         = float(request.args.get('Sixth_SEM'))
    Fifth_SEM         = float(request.args.get('Fifth_SEM'))
    Final_Performance = float(request.args.get('Final_Performance'))
    Medium            = int(request.args.get('Medium'))
    model1            = float(request.args.get('model1'))
    
    if model1==0:
      model=pickle.load(open('education_linearreg.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==1:
      model=pickle.load(open('education_Logisticreg.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==2:
      model=pickle.load(open('education_Decision_Tree.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==3:
      model=pickle.load(open('education_kernal_svm.pkl', 'rb')) 
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==4:
      model=pickle.load(open('education_linear_svm.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==5:
      model=pickle.load(open('education_randomforest.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==6:
      model=pickle.load(open('education_KNN.pkl', 'rb')) 
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==7:
      model=pickle.load(open('education_nb.pkl', 'rb'))
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))
    elif model1==8:
      model=pickle.load(open('education_kmeanscluster.pkl', 'rb'))      
      prediction = model.predict(sc.transform([[Tenth, Twelfth, B_Tech, Seventh_SEM, Sixth_SEM, Fifth_SEM, Final_Performance, Medium]]))

    if prediction==0:
      output="Student is Not Placed"
    else:
      output="Student is Placed"
        
    return render_template('index.html', prediction_text='Model  has predicted : {}'.format(output))      

if __name__=="__main__":
  app.run()
