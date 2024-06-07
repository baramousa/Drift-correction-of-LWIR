# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:30:19 2023
A script for correcting thermal drift using the Nearesr Neighbor algorithm
The input are tie point data exported from Agisoft using the agisoft_py_export_tie_points.py
@author: Albara
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn import neighbors

#path to the project folder
path1='C:\\Projects\\FOR2432\\A_WP3_model\\Agisoft\\'  
os.chdir(path1)
#path to the uncorrected thermal images
path_pics_in="C:\\Projects\\FOR2432\\A_WP2_on-station\\IR_drone\\pics\\231121_ir\\IR_4mosaic_231121\\IR_MOSAIC\\"
#path to the folder where corrected images will be saved
path_pics_out="C:\\Projects\\FOR2432\\A_WP2_on-station\\IR_drone\\pics\\231121_ir\\IR_4mosaic_231121\\IR_thermal mosaic_231121_knn_long\\"
#save the names of the uncorrected images in a list
pics=os.listdir(path_pics_in)

# open the text file of the tie points and save them in a data frame
data=pd.read_csv(path1+'IR_231121_ties_pionts.txt',decimal=".",sep="\t",header=None) 
#rename the data frame columns
data.rename(columns={0: 'camera', 1: 'x',2: 'y',3: 'temp',4: 'tie_point',5: 'n_proj',6:'err_pix'}, inplace=True)


#extract the time stamp from the images names and add it to the data frame
time_string=[]
for i in data.index:
    string=data['camera'][i].split("-")[0]+':'+data['camera'][i].split("-")[1]+':'+data['camera'][i].split("-")[2]
    time_string.append(datetime.strptime(string, '%H:%M:%S').time())
data['time']=time_string
# define the time stamp of the first and last images
first_photo_time= datetime.strptime("09:29:12", '%H:%M:%S').time()  
last_photo_time= datetime.strptime("09:36:18", '%H:%M:%S').time()  
#filter the data frame accordingly
data=data[(data['time']>=first_photo_time) & (data['time']<=last_photo_time)]

#A function to get the time in seconds starting with zero for the first image
def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

pics_zero=pics[0]
pics_zero=pics_zero.split("-")[0]+":"+pics_zero.split("-")[1]+":"+pics_zero.split("-")[2]
zero_time=get_sec(str(pics_zero))

time_sec=[]
for i in data.index:
    time_sec.append(get_sec(str(data.time[i]))-zero_time)
        
data['time_sec']=time_sec

# A function to assign swath or 'flight lane' number, the sec values are to be 
#defined by the user based on their own flight
def add_swath(df):
    
    df['swath']=-1

    df.loc[(df['time_sec']>=0) & (df['time_sec']<=35),['swath']]=1
    df.loc[(df['time_sec']>=40) & (df['time_sec']<=72),['swath']]=2
    df.loc[(df['time_sec']>=76) & (df['time_sec']<=108),['swath']]=3
    df.loc[(df['time_sec']>=112) & (df['time_sec']<=144),['swath']]=4
    df.loc[(df['time_sec']>=148) & (df['time_sec']<=180),['swath']]=5
    df.loc[(df['time_sec']>=183) & (df['time_sec']<=215),['swath']]=6
    df.loc[(df['time_sec']>=218) & (df['time_sec']<=250),['swath']]=7
    df.loc[(df['time_sec']>=254) & (df['time_sec']<=286),['swath']]=8
    df.loc[(df['time_sec']>=289) & (df['time_sec']<=321),['swath']]=9
    df.loc[(df['time_sec']>=324) & (df['time_sec']<=356),['swath']]=10
    df.loc[(df['time_sec']>=359) & (df['time_sec']<=391),['swath']]=11
    df.loc[(df['time_sec']>=394) & (df['time_sec']<=426),['swath']]=12
    df.drop(df[df['swath']==-1].index, inplace = True)
    
    return df
#Add the swath 'Lane' to the data frame
add_swath(data)
#filter the data frame excluding tie points with only one projection
df=data[(data['n_proj']>=1)&(data['time_sec']>=0)]

# loop through the swaths for forward drift correction

for k in range(2,12):
    #get the time in sec for the images in the k swath
    swath_list=df[df.swath==k].time_sec.tolist()
    #some exceptions regarding missing tie points in some images in some swaths  
    # sort the time in sec in ascending order
    swath_list.sort()
    # create a loop list accordingly
    loop_list=list(dict.fromkeys(swath_list))
    
    # loop through the list correcting forward drift in the k swath
    for i in loop_list:
    #for i in range(345,356):
        
        j=i+2
        # get a data frame with tie points of two successive images
        df_f=df[(df['time_sec']<j) & (df['time_sec']>=i)]
        
        #get the tie points that are common between image(i) and image(j), but only keep those in image(j)
        df_f_2= df_f[df_f['tie_point'].duplicated(keep = False) == True]
        # get the list of the common tie points ids
        grouped = df_f_2.groupby('tie_point')
        tie_ids=list(grouped.groups.keys())
        # calculate the drift with a condition the the number of tie points is larger than 25
        if len(tie_ids)>=25:

            delta_t=[]
            delta_temp=[]
            for ids in tie_ids:
                
                df_f=df_f_2[df_f_2['tie_point']==ids]
                delta_t.append(0)
                delta_temp.append(0)
                for index in df_f.index:
                    if index < df_f.index[-1]:
                        delta_t.append(df_f.time_sec[index+1]-df_f.time_sec[index])
                        delta_temp.append(df_f.temp[index+1]-df_f.temp[index])
            # add the drift to the data frame
            df_f_2.loc[:,'delta_temp']=delta_temp
            
            df_f_2= df_f_2[df_f_2['tie_point'].duplicated(keep = 'first') == True]
    
            # get the x-pixel and y-pixel coordinate of the tie points to be used as a fitting parameter
            X = df_f_2.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time', 'delta_temp','err_pix','time_sec','swath'], axis=1)
            X.shape
            # get the drift value of the tie points to be used as a target
            y = df_f_2['delta_temp']
            y.shape
            
            # split the set of X and Y to training and testing sets
            X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.8)
            #use the nearest neighbour algorithm for fitting
            knn = neighbors.KNeighborsRegressor(20, weights='distance')        
            knn.fit(X1.to_numpy(), y1.values)
    
            knn.score(X2.to_numpy(), y2.values)
                        
            # get the data frame of all the tie points that lay in image(i+1)
            df_tie_pts_cur=df[df.time_sec==i+1]
            # extract x and y of those tie points
            X_pred = df_tie_pts_cur.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time','err_pix','time_sec','swath'], axis=1)
            #predict x and y based on fitted nearest neighbour function
            y_pred = knn.predict(X_pred.values)
            
            #get the indecies of those points so they can be used to adjust tie point temp in the big data frame
            tie_pts_cur_ix=list(df_tie_pts_cur.index)
            # add the corrected temperature values based on the estimated drift from the nearest neighbour
            df.loc[tie_pts_cur_ix,'temp']=df_tie_pts_cur.temp-y_pred
            
    ###############################################################################
            # correct the image(i+1) and save the corrected image in the desired path
            camera=pics[i+1]
            #print(camera)
            filename = os.path.join(path_pics_in,camera)
            tst = io.imread(filename)

            #get the corresponding time
            xfit=[]
            x_y_temp=[]
            for idx, x in np.ndenumerate(tst):
                xfit.append([idx[1],idx[0]])
                x_y_temp.append([idx[1],idx[0], x/40-100])
            
            Xfit=np.array(xfit) 
            x_y_temp=np.array(x_y_temp)
            df_x_y_temp=pd.DataFrame(x_y_temp)
            pred=knn.predict(Xfit) 
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
            out = os.path.join(path_pics_out,camera)    
            # save the correct photo in the path    
            io.imsave(out,df_mod_array)
            
            # get the X pred
            #correct it 
            #save it 
            
        else:
            # an exception when the number of common tie points are less than 25, the correction is based on the nearest neighbour from the previous images
            df_tie_pts_cur=df[df.time_sec==i+1]
            if len(df_tie_pts_cur)!=0:
            # extract x and y of those tie points
                X_pred = df_tie_pts_cur.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time','err_pix','time_sec','swath'], axis=1)
                #predict x and y based on Knn fuction
                y_pred = knn.predict(X_pred.values)
                #y_pred = model.predict(X_pred.values)
                #get the indecies of those points so they can be used to adjust tie point temp in the big data frame
                tie_pts_cur_ix=list(df_tie_pts_cur.index)
                
                df.loc[tie_pts_cur_ix,'temp']=df_tie_pts_cur.temp-y_pred
                
                
                camera=pics[i+1]
                #print(camera)
                filename = os.path.join(path_pics_in,camera)
                tst = io.imread(filename)
                #get the corresponding time
                xfit=[]
                x_y_temp=[]
                for idx, x in np.ndenumerate(tst):
                    xfit.append([idx[1],idx[0]])
                    x_y_temp.append([idx[1],idx[0], x/40-100])
                
                Xfit=np.array(xfit) 
                x_y_temp=np.array(x_y_temp)
                df_x_y_temp=pd.DataFrame(x_y_temp)
                pred=knn.predict(Xfit) 
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
                out = os.path.join(path_pics_out,camera)    
                # save the correct photo in the path    
                io.imsave(out,df_mod_array)
            else:
                
                camera=pics[i+1]
                filename = os.path.join(path_pics_in,camera)
                tst = io.imread(filename)
                #get the corresponding time
                xfit=[]
                x_y_temp=[]
                for idx, x in np.ndenumerate(tst):
                    xfit.append([idx[1],idx[0]])
                    x_y_temp.append([idx[1],idx[0], x/40-100])
                
                Xfit=np.array(xfit) 
                x_y_temp=np.array(x_y_temp)
                df_x_y_temp=pd.DataFrame(x_y_temp)
                pred=knn.predict(Xfit) 
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
                out = os.path.join(path_pics_out,camera)    
                # save the correct photo in the path    
                io.imsave(out,df_mod_array)
                ################################
                continue
        

# the new tie points data frame based on forward drift correction is saved and used later for side drift correction
df.to_csv(path1+'IR231121_tie_points_corrected_knn_forward_drift.csv')
    
