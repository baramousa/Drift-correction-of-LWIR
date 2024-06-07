# -*- coding: utf-8 -*-
"""
This is a script for side thermal drift correction. Pre requiste ist to run 'swath_correction_optimized_code_step1_forward_drift.py', which run a forward drift correction 
First estimate the path to the folder to be corrected as well as the output folder

Created on Thu Aug 31 13:31:18 2023

@author: Albara
"""
import numpy as np
import matplotlib.pyplot as plt
#import datetime
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn import neighbors

#path to the project folder
path1='C:\\Projects\\FOR2432\\A_WP3_model\\Agisoft\\'
# path to the uncorrected thermal images
path_pics_in_or="C:\\Projects\\FOR2432\\A_WP2_on-station\\IR_drone\\pics\\231121_ir\\IR_4mosaic_231121\\IR_MOSAIC\\"
# path to the forward drift corrected images
path_pics_in="C:\\Projects\\FOR2432\\A_WP2_on-station\\IR_drone\\pics\\231121_ir\\IR_4mosaic_231121\\IR_thermal mosaic_231121_knn_long\\"
#path to the folder where side drift corrected images will be saved
path_pics_out="C:\\Projects\\FOR2432\\A_WP2_on-station\\IR_drone\\pics\\231121_ir\\IR_4mosaic_231121\\IR_thermal mosaic_231121_knn_long_side\\"

#open post forward drift corrected csv tie points file and save it as a data frame 
data=pd.read_csv(path1+'IR231121_tie_points_corrected_knn_forward_drift.csv',decimal=".",sep=",")
#drop first duplicate column of the data frame
data = data.drop(data.columns[[0]], axis=1)
#save the images names in a list
pics=os.listdir(path_pics_in_or)
#create a data frame of the images names
df_pics=pd.DataFrame(pics,columns =['camera'])

#extract the time in sec from the name of the image
time_string=[]
for i in df_pics.index:
    string=df_pics['camera'][i].split("-")[0]+':'+df_pics['camera'][i].split("-")[1]+':'+df_pics['camera'][i].split("-")[2]
    time_string.append(datetime.strptime(string, '%H:%M:%S').time())
    
df_pics['time']=time_string

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


zero_time=get_sec(str(df_pics.time[0]))


time_sec=[]
for i in df_pics.index:
    time_sec.append(get_sec(str(df_pics.time[i]))-zero_time)
    
    
df_pics['time_sec']=time_sec
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
df_pics=add_swath(df_pics)
          
df1=data.copy()
df1=add_swath(df1)
nswath=df_pics.swath.max()

# A function to get the common tie points between two parallel images
def get_common_ties(df1,t1,t2):
        
    df_f=df1[(df1['time_sec']==t1) | (df1['time_sec']==t2)]
    #get the tie points that are common between image0 and image1, but only keep those in image1
    df_f_2= df_f[df_f['tie_point'].duplicated(keep = False) == True]
    df_f_2

    grouped = df_f_2.groupby('tie_point')
    tie_ids=list(grouped.groups.keys())
    delta_t=[]
    delta_temp=[]
    for ids in tie_ids:
        #ids=tie_ids[20]
        df_f=df_f_2[df_f_2['tie_point']==ids]
        delta_t.append(0)
        delta_temp.append(0) 
        delta_temp.append(df_f.iloc[1,3]-df_f.iloc[0,3])
        
    df_f_2.loc[:,'delta_temp']=delta_temp    
    df_f_2= df_f_2[df_f_2['tie_point'].duplicated(keep = 'first') == True]
    return df_f_2
            
            
# loop through the swaths for side drift correction
for k in range(2,12):

    swath1=k
    swath2=k+1
    #get the time is seconds for the images is swath k
    df_time_1=df_pics[df_pics.swath==swath1].time_sec.tolist()
    df_time_1.sort()
    #get the time is seconds for the images is swath k+1
    df_time_2=df_pics[df_pics.swath==swath2].time_sec.tolist()
    df_time_2.sort()

    a = list(reversed(list(dict.fromkeys(df_time_1))))
    b=list(dict.fromkeys(df_time_2))
    
    loop_list=list(range(0,33))
    #loop through the images in the lanes
    for i in loop_list:
        #time stamp in sec of the i impage in swath k
        t1=a[i]
        #time stamp in sec of the i impage in swath k+1
        t2=b[i]
        #get the common tie points between those images
        df_f_2=get_common_ties(df1, t1, t2)
        
        # calculate the drift with a condition the the number of tie points is larger than 25
        if len(df_f_2)>25:
            
            # get the x-pixel and y-pixel coordinate of the tie points to be used as a fitting parameter
            X = df_f_2.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time', 'delta_temp','err_pix','time_sec','swath'], axis=1)
            X.shape
            # get the drift value of the tie points to be used as a target
            y = df_f_2['delta_temp']
            y.shape
            
            # divide the data to training and validation sets
            X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.8)
            knn = neighbors.KNeighborsRegressor(20, weights='distance')      
            knn.fit(X1.to_numpy(), y1.values)

            knn.score(X2.to_numpy(), y2.values)
            
            # get the data frame of all the tie points that lay in image i+1
            df_tie_pts_cur=df1[df1.time_sec==t2]
            # extract x and y of those tie points
            X_pred = df_tie_pts_cur.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time','err_pix','time_sec','swath'], axis=1)
            #predict x and y based on Knn fuction
            y_pred = knn.predict(X_pred.values)
            #get the indecies of those points so they can be used to adjust tie point temp in the big data frame
            tie_pts_cur_ix=list(df_tie_pts_cur.index)
            # correct the tie point temp based on the function
            df1.loc[tie_pts_cur_ix,'temp']=df_tie_pts_cur.temp-y_pred
            #open the camera/photo for correcting all pixels based on the function
            
            camera=pics[t2]
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
            # In the case the number of tie points is less than 25, the correction is done based on the previous function from the previous
            df_tie_pts_cur=df1[df1.time_sec==t2]
            df_tie_pts_cur
            #Here the condtion is for checking if there are any tie points at all in that photo, in case yes, they should be corrected, so
            # they are ready to use when compared with pictures from the following swath
            if len(df_tie_pts_cur)!=0:
            # extract x and y of those tie points
                X_pred = df_tie_pts_cur.drop(['camera', 'temp', 'tie_point', 'n_proj', 'time','err_pix','time_sec','swath'], axis=1)
                #predict x and y based on Knn fuction
                y_pred = knn.predict(X_pred.values)
                #y_pred = model.predict(X_pred.values)
                #get the indecies of those points so they can be used to adjust tie point temp in the big data frame
                tie_pts_cur_ix=list(df_tie_pts_cur.index)
                
                df1.loc[tie_pts_cur_ix,'temp']=df_tie_pts_cur.temp-y_pred
                            
#######################################################################################
##########################################################################################
#########################################################################################
                
                camera=pics[t2]
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
                # the last condition is for pictures that have no tie points at all
                camera=pics[t2]
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
                ###################################################################
                ##################################################################
                continue
            
    
df2=df1.copy()
#save the data frame with corrected drift
df2.to_csv(path1+'IR231121_tie_points_corrected_knn_forward_side.csv')
