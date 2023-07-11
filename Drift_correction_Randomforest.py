# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:30:19 2023
A script for correction of thermal drift using random forest regression
The input are tie point data exported from Agisoft using the agisoft_py_export_tie_points.py
@author: Albara
"""



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,ShuffleSplit
from skimage import io

path1='C:\\Projects\\Agisoft\\' 
os.chdir(path1)
curdir = os.getcwd()

data=pd.read_csv(path1+'IR_261121_ties_Gh.txt',decimal=".",sep="\t",header=None) # IR_261121_ties.txt,, tie_points_with_pxerror.txt
data.rename(columns={0: 'camera', 1: 'x',2: 'y',3: 'temp',4: 'tie_point',5: 'n_proj',6:'err_pix'}, inplace=True)
time_string=[]
# extract the time from the camera's name
for i in data.index:
    string=data['camera'][i].split("-")[0]+':'+data['camera'][i].split("-")[1]+':'+data['camera'][i].split("-")[2]
    time_string.append(datetime.strptime(string, '%H:%M:%S').time())


        
data['time']=time_string

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)



zero_time=get_sec(str(data.time[0]))


time_sec=[]
# convert the time to seconds
for i in data.index:
    time_sec.append(get_sec(str(data.time[i]))-zero_time)
    
    
data['time_sec']=time_sec

# filter out tie points with less than 4 projections (i.e: appearing in less than ten photos)
df=data[data['n_proj']>10]

grouped = df.groupby('tie_point')
tie_ids=list(grouped.groups.keys())
delta_t=[]
delta_temp=[]
# get delta temp and corresponding delta time of the tie points between to consecutive photos 
for ids in tie_ids:
    df_f=df[df['tie_point']==ids]
    delta_t.append(0)
    delta_temp.append(0)

    for index in df_f.index:
        if index < df_f.index[-1]:
            delta_t.append(df_f.time_sec[index+1]-df_f.time_sec[index])
            delta_temp.append(df_f.temp[index+1]-df_f.temp[index])
df['delta_t']=delta_t
df['delta_temp']=delta_temp
# get delta temp of tie points relative to the first photo they apeared in
delta_temp_sum=[]
for ids in tie_ids:
    df_f=df[df['tie_point']==ids]
    delta_temp_sum.append(0)
    for index in df_f.index:
        if index < df_f.index[-1]:
            delta_temp_sum.append(df_f.delta_temp[index+1]+delta_temp_sum[-1])
        
df['delta_temp_sum']=delta_temp_sum
    
 
    
# filter out tie points with projection error higher than 10 pixels
df_f= df[df['err_pix']<10]
# check the number of resulting tie points used for drift correction
grouped = df_f.groupby('tie_point')
print(grouped.ngroups) 


# Sorting the features matrix
X = df_f.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time', 'delta_t', 'delta_temp','delta_temp_sum','err_pix'], axis=1)
X.shape


# Target array
y = df_f['delta_temp_sum']
y.shape
    

###############################################################################
###############################################################################
###########################Random Forest regression############################
###############################################################################
###############################################################################
forest = RandomForestRegressor(200,n_jobs=8)#max_depth, ,min_samples_leaf=i

## First:cross validation with ShuffleSplit method , a method that randomise the cv

n_samples = X.shape[0]
# basically it divide the data seven tives, each time it fits 70% of the data and test it on the rest 30%
cv = ShuffleSplit(n_splits=7, test_size=0.3, random_state=1)
#run the cross validation, output the score insdie an array
cross_val_score(forest, X.to_numpy(), y.values, cv=cv)

# now get the model based on 80% of the data
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.8)

# fit the model on one set of data
forest.fit(X1.to_numpy(), y1.values)

# evaluate the model on the second set of data (the 20% of the data)

forest.score(X2.to_numpy(), y2.values)

###############################################################################
###############################################################################
###################################Drift correction############################
###############################################################################
path_pics_in='C:\\Projects\\in\\'
path_pics_out='C:\\Projects\\out\\'

pics=os.listdir(path_pics_in)

pics_time=[]
for i in range(len(pics)):
    string=pics[i].split('-')[0]+':'+pics[i].split('-')[1]+':'+pics[i].split('-')[2]
    pics_time.append(datetime.strptime(string, '%H:%M:%S').time())


zero_time=get_sec(str(pics_time[0]))


time_sec=[]
for i in pics_time:
    time_sec.append(get_sec(str(i))-zero_time)
    

# dictionary of the pic and it's corresponding time in sec
pics_sec=dict(zip(pics, time_sec))


for i in range(len(pics)):
    #i=1
    #path to photo
    filename = os.path.join(path_pics_in,pics[i])
    #read the photo as numpy array
    tst = io.imread(filename)
    #get the corresponding time
    pic_time=pics_sec[pics[i]]
    xfit=[]
    x_y_temp=[]
    for idx, x in np.ndenumerate(tst):
        xfit.append([idx[1],idx[0],pic_time])
        x_y_temp.append([idx[1],idx[0], x/40-100])
    #features matrix
    Xfit=np.array(xfit) 
    x_y_temp=np.array(x_y_temp)
    df_x_y_temp=pd.DataFrame(x_y_temp)
    #predict the drift according to the features
    pred=forest.predict(Xfit) 
    df_x_y_temp['drift']=pred
    #correct the temp according to the drift
    df_x_y_temp['temp_mod']=df_x_y_temp[2]-df_x_y_temp['drift']
    df_x_y_temp=df_x_y_temp.drop('drift',axis=1)
    df_x_y_temp=df_x_y_temp.drop(2,axis=1)
    df_x_y_temp['temp_mod']=(df_x_y_temp['temp_mod']+100)*40
    df_x_y_temp=df_x_y_temp.drop(0,axis=1)
    df_x_y_temp=df_x_y_temp.drop(1,axis=1)
    df_mod_array=df_x_y_temp.to_numpy()
    df_mod_array=df_mod_array.reshape((512, 640))
    
    out = os.path.join(path_pics_out,pics[i])    
    # save the correct photo in the path    
    io.imsave(out,df_mod_array)

