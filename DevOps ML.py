#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from flask import Flask, request, jsonify
import traceback
import numpy as np
import sqlite3

Qestion 1.

# Use the training_data.csv to build and train a simple classification model. 

# The model needs to predict the target using the features f1, f2, and f3

# Simple Modeling without paramter Tuning 
def DevOps_ML(df):
    # Read Data
    data = pd.read_csv(df)
    # check missing value per each column
    count_nan = len(data) - data.count()
    #print(count_nan)
    # Remove the missing value on feature column f3 and store back to Data
    data = data.dropna()
    data  =  pd.get_dummies(pd.DataFrame(data))
    # Label And Features 
    labels = data['target']
    features = data.drop('target', axis=1)
    
    # Split the data to train test and validation
    train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)
    
    RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10)
    model =  RF.fit(train, train_labels)
    preds = RF.predict(test)
    
    Accuracy = accuracy_score(test_labels, preds)
    print ("The closser to 1 the better, The Accuracy score of the Model is {} ".format(Accuracy))
    
    joblib.dump(model, 'model.pkl')
    print("Model dumped!")
    
    # Saving the data columns from training
    model_columns = list(features.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("Models columns dumped!")
    
    return model 
        
DevOps_ML('training_data.csv')   


#------------------------------------------------------------------------------------------------

# Load Model 

def model_saving(model):
    
    # Load the model that I just saved
    RF = joblib.load(model)
    return RF

RF = model_saving('model.pkl')

#----------------------------------------------------------------------------------------------

# SQLite database to log and monitor the model performance.

# Create a Database if not exist

def connectDB():
    file = []
    for filename in os.listdir():
        file.append(filename)
    if 'sqn.db' in file:
        return 'Database already exist'
    else:
        conn = sqlite3.connect('sqn.db')
        print ("Opened database successfully")
    
        # Create Tables
        conn.execute('CREATE TABLE classification_request (id_request INTEGER PRIMARY KEY AUTOINCREMENT, request_timestamp TEXT NOT NULL, predicted_class INTEGER,response_status TEXT NOT NULL,error_message TEXT)')
        conn.execute('CREATE TABLE classification_request_param (id_request INTEGER PRIMARY KEY AUTOINCREMENT,f1 TEXT NOT NULL,f2 TEXT NOT NULL,f3 TEXT NOT NULL, FOREIGN KEY(id_request) REFERENCES classification_request(id_request))')
        print ("Table created successfully")
        conn.close()

connectDB()
    
 #------------------------------------------------------------------------------------------------   

# Flask application with a REST API
# Flask application with a REST API

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import logging
import traceback
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import pytz
from collections import Counter
from statistics import mean 


# API definition
app = Flask(__name__)

# End point Classify
@app.route('/classify', methods=['POST', 'GET'])
def predict():
    if rf:
        try:
            json_ = request.json
            print(json_)
            f1 = json_['f1']
            f2 = json_['f2']
            f3 = str(json_['f3'])
            
            
            query = pd.get_dummies(pd.DataFrame(json_ , index=[0]))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = rf.predict(query)
            prediction_res = int(prediction[0])
            
            tz = pytz.timezone('Europe/Berlin')
            request_timestamp = datetime.now(tz)
            
           # Request logged into Database if certain condition are met, as specified in the question
            
            if request.method == 'POST':
                try:
                    with sqlite3.connect("sqn.db") as con:
                        cur = con.cursor()
                        print('consor connected')

                        if type(f1) != float:
                            response_status = "ERROR"
                            error_message = "f1 is not a float"
                            prediction_res = None
                            cur.execute("INSERT INTO classification_request(request_timestamp, predicted_class,response_status,error_message ) VALUES(?,?,?,?)",(request_timestamp, prediction_res, response_status,error_message) )
                            cur.execute("INSERT INTO classification_request_param(f1,f2,f3) VALUES(?,?,?)",(f1,f2,f3) )
                            con.commit()
                            msg = "Record successfully added"
                            return jsonify({"status": "ERROR", "error_message": "f1 is not a float."})
                        
                        elif type(f2) != float:
                            print('skip2')
                            response_status = "ERROR"
                            error_message = "f2 is not a float"
                            prediction_res = None
                            cur.execute("INSERT INTO classification_request(request_timestamp, predicted_class,response_status,error_message ) VALUES(?,?,?,?)",(request_timestamp, prediction_res, response_status,error_message) )
                            cur.execute("INSERT INTO classification_request_param(f1,f2,f3) VALUES(?,?,?)",(f1,f2,f3) )
                            con.commit()
                            msg = "Record successfully added"
                            return jsonify({"status": "ERROR", "error_message": "f2 is not a float."})
                        
                        elif type(f3) != str:
                            print('skip3')
                            response_status = "ERROR"
                            error_message = "f3 is not a string"
                            prediction_res = None
                            cur.execute("INSERT INTO classification_request(request_timestamp, predicted_class,response_status,error_message ) VALUES(?,?,?,?)",(request_timestamp, prediction_res, response_status,error_message) )
                            cur.execute("INSERT INTO classification_request_param(f1,f2,f3) VALUES(?,?,?)",(f1,f2,f3) )
                            con.commit()
                            msg = "Record successfully added"
                            return jsonify({"status": "ERROR", "error_message": "f3 is not a string."})

                        lastrow = []
                        def sql_fetch(con):
                            cursorObj = con.cursor()
                            cursorObj.execute('SELECT * FROM classification_request_param WHERE id_request = (SELECT MAX(id_request)-1 FROM classification_request_param)')
                            rows = cursorObj.fetchall()
                            for row in rows:
                                for i in row:
                                    lastrow.append(i)
                        sql_fetch(con)
                        
                        #con.close()
                        
                        lastrow_com = lastrow[1:]
                        jsonincome = []
                        for j in json_:
                            jsonincome.append(json_[j])
                            
                        lastrow_f = []
                        for s in lastrow_com:
                            try:
                                lastrow_f.append(float(s))
                            except ValueError:
                                lastrow_f.append(s)
                        
                        if jsonincome == lastrow_f:
                            response_status = "WARNING"
                            error_message = None
                            cur.execute("INSERT INTO classification_request(request_timestamp, predicted_class,response_status,error_message ) VALUES(?,?,?,?)",(request_timestamp, prediction_res, response_status,error_message) )
                            cur.execute("INSERT INTO classification_request_param(f1,f2,f3) VALUES(?,?,?)",(f1,f2,f3) )
                            con.commit()
                            msg = "Record successfully added"
                            return jsonify({"predicted_class": int(prediction), "status": "WARNING"})
                            
                        elif (type(f1) != float) or (type(f2) != float) or (type(f3) != str) == True :
                            return 
                            
                        else:
                            response_status = "OK"
                            error_message = None
                            cur.execute("INSERT INTO classification_request(request_timestamp, predicted_class,response_status,error_message ) VALUES(?,?,?,?)",(request_timestamp, prediction_res, response_status,error_message) )
                            cur.execute("INSERT INTO classification_request_param(f1,f2,f3) VALUES(?,?,?)",(f1,f2,f3) )
                            con.commit()
                            msg = "Record successfully added"
                            return jsonify({'predicted_class': int(prediction),  "status": "OK" })
                            
                except:
                    con.rollback()
                    msg = "error in insert operation"
                finally:
                    print(msg)
                    con.close()
        except:
            
            return jsonify({'trace': traceback.format_exc()})

        return 'Test with Postman'
    
        
    else:
        print ('Train the model first')
        return ('No model here to use')
    
# End point /stats, return mean of f1, f2 and most frequent f3 
@app.route('/stats', methods=['POST', 'GET'])
def stat():
    if rf:
        try:
            mean_f1_list = []
            mean_f2_list = []
            mostFrequent_f3_list = []
            if request.method == 'GET':
                try:
                    with sqlite3.connect("sqn.db") as con:
                        cur = con.cursor()
                        print('consor connected')
                        cur.execute("""SELECT c.id_request, cast(c.f1 as INT), cast(c.f2 as INT), c.f3
                                    FROM classification_request_param c 
                                    INNER JOIN
                                    classification_request p
                                    ON c.id_request = p.id_request
                                    WHERE  p.response_status = "OK" ; """)
                        print('execute ok')
                        rows = cur.fetchall()
                        print('fetch ok')
                        for row in rows:
                            mean_f1_list.append(row[1])
                            mean_f2_list.append(row[2])
                            mostFrequent_f3_list.append(row[3])
                    sql_fetch(con)
                    print('fetch ok')
                    mean_f1 =  mean(mean_f1_list)
                    mean_f2 =  mean(mean_f2_list)
            
                    def most_frequent(List): 
                        occurence_count = Counter(List) 
                        return occurence_count.most_common(1)[0][0] 
                    f3_most_frequent = most_frequent(mostFrequent_f3_list)
            
                    print(mean_f1)
                
                except:
                    con.rollback()
                    msg = "error in insert operation"
                finally:
                    #print(msg)
                    con.close()
        
        except:
            return jsonify({'trace': traceback.format_exc()})
        
        print(mean_f1)
        print(mean_f2)
        print(f3_most_frequent)
        
        return jsonify({"mean_f1": mean_f1, "mean_f2": mean_f2, "most_frequent_f3": f3_most_frequent})
    else:
        print ('Train the model first')
        return ('No model here to use')
    

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 1234 # If you don't provide any port the port will be set to 12345

    rf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=5000, debug=False)




