import pandas as pd
import numpy as np
df = pd.read_csv('data/heart.csv')
#lập bảng tóm lược dữ liệu
def laopBangTomLuoDuLieu(df):
    print(df[['Patient ID','Age','Sex','Cholesterol','Blood Pressure',
            'Heart Rate','Diabetes','Family History','Smoking',
            'Obesity','Alcohol Consumption','Exercise Hours Per Week',
            'Diet']].describe())
    print(df[['Previous Heart Problems','Medication Use','Stress Level',
            'Sedentary Hours Per Day','Income','BMI','Triglycerides',
            'Physical Activity Days Per Week','Sleep Hours Per Day',''
            'Country','Continent','Hemisphere','Heart Attack Risk'
    ]].describe())
# Kiểm tra giữ liệu thiếu
def khaoSatDulieuThieu(df):
    print("Missing values:\n", df.isnull().sum())
#Kiểm tra dữ liệu trùng
def khaoSatDuLieuTrung(df):
    print("Duplicate values:\n", df.duplicated().sum())
#Loại bỏ dữ liệu thiếu
def loaiBoDuLieuThieu(df):
    df.dropna(inplace=True)
#Chuyển đổi categorical variables
def chuyenDoiCotSex(df):
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
#Chuyển đổi categorical variables
def chuyenDoiCotDiet(df):
    df['Diet'] = df['Diet'].map({'Healthy': 2, 'Average': 1, 'Unhealthy': 0})
#Tách cột Blood Pressure thành Systolic và Diastolic
def tachCotBloodPressure(df):
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df.drop('Blood Pressure', axis=1, inplace=True)
# Chuẩn hóa dữ liệu số
def chuanHoaDuLieuSo(df):
    numeric_columns = ['Age', 'Cholesterol', 'Heart Rate', 'BMI', 'Triglycerides']
    for col in numeric_columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
#Kiểm tra phân phối của target variable
def kiemTraPhanPhoiCuaTargetVariable(df):
    print("Target distribution:\n", df['Heart Attack Risk'].value_counts(normalize=True))
