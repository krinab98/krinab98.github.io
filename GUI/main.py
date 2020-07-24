import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
model=pickle.load(file)
file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form

        FVC = float(myDict['FVC'])
        Haemoptysis = float(myDict['Haemoptysis'])
        Performance = int(myDict['Performance'])
        Dyspnoea=int(myDict['Dyspnoea'])
    #     # Pain = int(myDict['Pain'])
        Cough = int(myDict['Cough'])
    #     # Weakness = int(myDict['Weakness'])
        Tumor = int(myDict['Tumor'])
        MI_6mo = int(myDict['MI_6mo'])
        Smoking = int(myDict['Smoking'])
        Asthama = int(myDict['Asthama'])
        AGE = int(myDict['AGE'])


        print(request.form)
        
        input=[FVC,Haemoptysis,Performance,Dyspnoea,Cough,Tumor,MI_6mo,Smoking,Asthama,AGE]
        # input_features[0]=FVC
        # input_features[1]=Haemoptysis
        # input_features[2]=Performance
        # input_features[3]=Dyspnoea
        # input_features[4]=Cough
        # input_features[5]=Tumor
        # input_features[6]=MI_6mo
        # input_features[7]=Smoking
        # input_features[8]=Asthama
        # input_features[9]=AGE

        input_features=np.array(input).reshape(1,-1)
        infProb=model.predict_proba(input_features)[0][0]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    #return 'Hello, World!' 
if __name__ == "__main__":
    app.run(debug=True)

