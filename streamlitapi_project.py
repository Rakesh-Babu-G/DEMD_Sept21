#from typing import no_type_check
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
from statistics import mode


def preprocess(user_ip):
    '''
    # user input preprocessing module
    # one hot encoding of the categorical variable 'Stream'
    # returning the required columns for ML model input
    '''
    user_ip_df = pd.DataFrame([user_ip], columns = ['Age','Internships', 'CGPA', 'Hostel','Stream'])
    train_categories = ['Civil', 'Computer Science',  'Electrical',  'Electronics And Communication',
                         'Information Technology',  'Mechanical'] # from training set
    stream_catg = pd.Categorical(user_ip_df['Stream'], categories = train_categories)
    one_hot_df = pd.get_dummies(stream_catg)
    concat_df = pd.concat([user_ip_df,one_hot_df], axis=1)
    concat_df.drop(['Stream'], axis= 1, inplace=True)
    
    return concat_df


def main_prediction():
    '''
    # main prediction function
    # accepts input from user & predicts if the student gets placed or not
    '''
    # title of the page
    st.title("Student Placement Prediction - ML API")
        
    # user inputs
    Age = st.number_input("Age",19,35)
    Internships = st.number_input("Internships",0,10)
    CGPA = st.number_input("CGPA",0,10)
    Hostel = st.number_input("Hostel",0,1)
    #Stream = st.text_input("Stream")
    Stream = st.selectbox("Stream",
    ('Civil', 'Computer Science',  'Electrical',  'Electronics And Communication','Information Technology', 'Mechanical'))

    # user input preprocesing for ML model input
    user_ip = [Age,Internships,CGPA,Hostel,Stream]
    model_ip = preprocess(user_ip)

    # pickle files of the models
    pickled_model1 = open("Pickle_file_Placement_lr.pkl", "rb")
    classifier1 = pickle.load(pickled_model1)
    pickled_model2 = open("Pickle_file_Placement_dt.pkl", "rb")
    classifier2 = pickle.load(pickled_model2)
    pickled_model3 = open("Pickle_file_Placement_knn.pkl", "rb")
    classifier3 = pickle.load(pickled_model3)
    
    # ensemble of 3 models & mode of the predictions is selected as the prediction
    result = mode( [ classifier1.predict(model_ip)[0],
                     classifier2.predict(model_ip)[0],
                     classifier3.predict(model_ip)[0] ] )
    if result == 1:
        prediction = 'Gets placed'
    else:
        prediction = 'Not placed'

    # button print result
    if st.button("prediction"):
        st.success(f"Student {prediction}")        

if __name__ == "__main__":
    main_prediction()

# ==============================================================================
# ==============================================================================
