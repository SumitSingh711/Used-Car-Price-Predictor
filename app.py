import pickle

import warnings

import joblib

import pandas as pd

import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# sklearn.set_config(transform_output='pandas') # To show all output to pandas
# convenience functions

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year', 'kms_driven']),  # Apply StandardScaler to numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])  # Apply OneHotEncoder to categorical columns
    ])


# Create a pipeline that includes preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# read the training data
car = pd.read_csv("Data/cleaned_car_data.csv")

X = car.drop(columns="price")
y = car.price.copy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

# fit and save the model pipeline
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'pipeline.joblib')


# web application
st.set_page_config(
    page_title='Used Car Price Prediction',
    page_icon='🚘'
)

st.title('Used Car Prediction Streamlit')


# user inputs

name = st.selectbox(
     'Model Name',
     options=X_train.name.unique()
)

# Create a number input for the year
year = st.number_input(
    "Model Year:",
    min_value=1900,  # Minimum year
    max_value=2024,  # Maximum year
    value=2024,      # Default value
    step=1           # Step value
)

kms_driven = st.number_input(
    'Odometer Reading (kms)',
    step=100
)

fuel_type = st.selectbox(
     'Fuel Type',
     options=X_train.fuel_type.unique()
)


x_new = pd.DataFrame(dict(
	name = [name],
	company=[company],
	year=[year],
	kms_driven=[kms_driven],
	fuel_type=[fuel_type]
)).astype({
	col: "str"
	for col in ["year"]
})


if st.button('Predict'):

    car_pipeline = joblib.load('pipeline.joblib')
    pred = car_pipeline.predict(x_new)[0]

    st.success(f"The predicted price is {pred:,.0f} INR")

