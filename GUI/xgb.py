import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix,accuracy_score
from xgboost import XGBClassifier
if __name__ == '__main__':
    df1=pd.read_csv('Final.csv')
    #Consider RFE as feature extracxtion, cosnider only that particular features
    df = df1.loc[:, ['PRE4','PRE6','PRE8','PRE9','PRE10','PRE14','PRE19','PRE30','PRE32','AGE','Risk1Yr']]
    X=df.iloc[:,:-1]
    y=df["Risk1Yr"].iloc[:]
    #Splitting Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=0)
    # Feature Scaling Data
    from sklearn.preprocessing import StandardScaler
    sc_X =StandardScaler()
    X_train =sc_X.fit_transform(X_train)
    X_test =sc_X.fit_transform(X_test)
    
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]

    # # evaluate predictions
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # #Predict the model
    # #predict the model
    # info_predict=model.predict(np.array([[3.46,2,1,1,1,17,1,1,1,3]]))
    # print(info_predict)

    # info_prob=model.predict_proba(np.array([[3.46,2,1,1,1,17,1,1,1,3]]))
    # print(info_prob)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(model, file)
    file.close()
    