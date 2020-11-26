#Vehicle Speed Prediction using Machine learning
# Useful Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,GRU
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xlwt
from scipy.stats import pearsonr
from keras.models import load_model
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.preprocessing import MinMaxScaler


# Input folder.
# Please enter the input folder path which has training excel files

def collect_data(dirName):
    listOfFile = os.listdir(dirName)

    allFiles = list()
    for entry in listOfFile:

        fullPath = os.path.join(dirName, entry)

        if os.path.isdir(fullPath):
            allFiles = allFiles + collect_data(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def Consolidating_dataframe(allFiles):
    print(allFiles)
    main = pd.DataFrame()
    for f in allFiles:
        if (f.endswith('.csv')):
            dt = pd.read_csv(f)
            main = main.append(dt)
    return main



def Data_Normalization(Features, Needed=True):
    # Normalization for features
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Output = Output.to_frame()
    if (Needed):
        Features_n = scaler.fit_transform(Features)
        Features_normalied = pd.DataFrame(Features_n, index=Features.index, columns=Features.columns)
    return Features_normalied


def Correlation_find(Features,Label,Needed = True):
    if(Needed):
        Correlation = (Features.corrwith(Output).abs())
        Correlation = Correlation.sort_values(ascending =  False)
        Correlation = Correlation.to_frame()
        return Correlation


def Read_Excel_data(D1, output_parameter):
    D1 = pd.read_csv(file_name)
    D1.drop(["Time"], axis=1)
    Label = pd.DataFrame()
    D1.drop("Time","AxVeh","VxVeh_Estimated","AyVeh","whlslip_RL","whlslip_RR","whlslip_FR","whlslip_FL","Ice","Snow","DryAsphalt")
    Label = D1[output_parameter]
    D1 = D1.drop([output_parameter, 'Time'], axis=1)

    return D1, Label


def Write_results_Excel(cols, Relation):
    wb = xlwt.Workbook()  # create empty workbook object
    newsheet = wb.add_sheet('Results')  # sheet name can not be longer than 32 characters
    for it in range(0, 2):
        for ele in range(0, len(cols)):
            if (it == 0):
                newsheet.write(ele, it, cols[ele])  # write contents to a cell, marked by row i, column j
            else:
                newsheet.write(ele, it, Relation[ele] * 100)


    wb.save('Correlation_results.xls')

def Data_processing(Features):
    Features['Vwhl_RL'] = Features['Vwhl_RL']*0.9857
    Features['Vwhl_RR'] = Features['Vwhl_RR']*0.9857
    Features['Vwhl_FR'] = Features['Vwhl_FR']*0.9857
    Features['Vwhl_FL'] = Features['Vwhl_FL']*0.9857
    Features['Avg_whl_spd'] = (Features['Vwhl_RL'] + Features['Vwhl_RR'] + Features['Vwhl_FR'] + Features['Vwhl_FL'])/4
    Features['Max_Whlspd']=Features[['Vwhl_RL','Vwhl_RR','Vwhl_FR','Vwhl_FL']].max(axis=1)
    Features['GPS_VxF'][Features['GPS_VxF'] < 0] = 0
    return Features


def Sequencing(x_data, y_data, num_steps):
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    num = 0
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        # if index is larger than the size of the dataset, we stop
        if end_ix >= x_data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = x_data.iloc[i:end_ix, :]
        seq_X = seq_X.to_numpy()
        # seq_X
        # Get only the last element of the sequency for y
        seq_y = y_data.iloc[end_ix, 0]  # y_data[end_ix][0]
        # Append the list with sequencies
        # print(type(seq_X))
        # print(type(seq_y))
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays

    # x_array = np.array(X)
    # y_array = np.array(y),
    # num = num + 1
    # print(num)
    return X, y


def ML_model(X_train, Y_train, num_steps, batch_s, epoch_n, train_model):
    Y_train = np.array(Y_train)
    print(Y_train.shape)
    # Y_train = Y_train.T
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    print((X_train.shape))
    print((Y_train.shape))
    initializer = keras.initializers.TruncatedNormal(mean=0, stddev=0.5)
    # Model architecture being followed.
    model = Sequential()
    model.add(GRU(12, activation='relu', kernel_initializer=initializer, return_sequences=True,input_shape=(num_steps, X_train.shape[2])))
    model.add(GRU(25, activation='relu', kernel_initializer=initializer, return_sequences=True))
    model.add(GRU(50, activation='relu', kernel_initializer=initializer, return_sequences=True))
    model.add(GRU(25, activation='relu', kernel_initializer=initializer, return_sequences=True))
    model.add(GRU(12, activation='relu', kernel_initializer=initializer))
    model.add(Dense(1, activation='relu', kernel_initializer=initializer))
    model.compile(optimizer='adam', loss=keras.losses.MeanAbsolutePercentageError(), metrics=["mean_squared_error"])
    model.summary()
    model.fit(X_train, Y_train, epochs=epoch_n, batch_size=batch_s)
    model.save(train_model)
    del model


def slice_joining(pre_Xtrain, pre_Ytrain, num_steps, slice_length):
    l1 = pre_Xtrain.shape[0]
    first = 0
    i = 0
    while (i < l1):
        first = first + 1
        if ((i + slice_length) > l1):
            if (i == 0):
                temp_X_train = pre_Xtrain[i:l1]
                temp_Y_train = pre_Ytrain[i:l1]
            else:
                temp_X_train = pre_Xtrain[i - num_steps:l1]
                temp_Y_train = pre_Ytrain[i - num_steps:l1]
            i = l1
        else:
            if (i == 0):
                temp_X_train = pre_Xtrain[i:i + slice_length]
                temp_Y_train = pre_Ytrain[i:i + slice_length]
            else:
                temp_X_train = pre_Xtrain[i - num_steps:i + slice_length]
                temp_Y_train = pre_Ytrain[i - num_steps:i + slice_length]
            i = i + slice_length
        tx_train, ty_train = Sequencing(temp_X_train, temp_Y_train, num_steps)
        ty_train = np.array(ty_train)
        ty_train = ty_train.reshape(ty_train.shape[0], 1)
        tx_train = np.array(tx_train)
        if (first == 1):
            X_train = tx_train
            Y_train = ty_train

        else:
            X_train = np.append(X_train, tx_train, axis=0)
            Y_train = np.append(Y_train, ty_train, axis=0)
        print(X_train.shape)
        print(Y_train.shape)

    return X_train, Y_train

def Test_model(X_test,Y_test,train_model):
    Predicted_values = []
    pred = []
    GT = []
    loaded_model=load_model(train_model)
    Predicted_values=loaded_model.predict(X_test).tolist()
    print(Predicted_values[:5])
    for ele in Predicted_values:
        pred.append(ele[0])
    #print(Y_test.head)
    print(Y_test)
    for ele in Y_test:
        GT.append(ele[0])
    plot_graph(pred,GT)


def plot_graph(pred, GT):
    time = list(range(1, len(GT) + 1))
    print(time[:5])
    fig = go.Figure()

    print(pred[:5])
    print(GT[:5])
    fig.add_trace(go.Scatter(x=time, y=GT, name='ground_truth'))
    fig.add_trace(go.Scatter(x=time, y=pred, name='Vx_predicted'))
    plot(fig, filename='GRU_road_version_1_2')

#main

Inp_folder='Give the input folder name'
slice_length = 1000
split_ratio = 80
num_steps = 10
batch_size = 1000
epoch = 75
train_model = 'baseline_trained_model_DA_carsim.h5'
print(Inp_folder)

df = pd.DataFrame()
file_num = collect_data(Inp_folder)
df = Consolidating_dataframe(file_num)
pre_features = df
pre_features = Data_processing(pre_features)
pre_features.fillna(0)
Features,Output = Read_Excel_data(pre_features,"GPS_VxF")

frame = { "GPS_VxF": Output}
Output = pd.DataFrame(frame)

mid = (int)((split_ratio*(Features.shape[0]))/100)

pre_Xtrain = Features[:mid]
pre_Ytrain = Output[:mid]
pre_Xtest = Features[mid:]
pre_Ytest = Output[mid:]

pre_Xtrain = pre_Xtrain[['Vwhl_RL','Vwhl_FR','Vwhl_RR','Vwhl_FL','Avg_whl_spd','Max_Whlspd','SAS','Brake_by_Driver']]
pre_Xtest = pre_Xtest[['Vwhl_RL','Vwhl_FR','Vwhl_RR','Vwhl_FL','Avg_whl_spd','Max_Whlspd','SAS','Brake_by_Driver']]

print("doing")
print(pre_Xtrain.shape)
print(pre_Ytrain.shape)

X_train,Y_train = slice_joining(pre_Xtrain,pre_Ytrain,num_steps,slice_length)
X_test,Y_test = slice_joining(pre_Xtest,pre_Ytest,num_steps,slice_length)
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)

train_model = 'baseline_trained_model_DA_carsim_diff_6epochs.h5'
Test_model(X_test,Y_test,train_model)
