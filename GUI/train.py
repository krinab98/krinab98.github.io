import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
 
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
    #Random Forest Modelling
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    oiut=classifier.predict_proba([[3.46,2,1,0,1,17,0,1,1,3]])
    print(oiut)