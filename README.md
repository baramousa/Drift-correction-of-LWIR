# Drift-correction-of-LWIR
Correction of thermal drift induced by changing core temperature and NU (non uniformity) of uncooled thermal cameras. 

Tie point are first extracted using the agisoft_py_export_tie_points.py file, after photos are aligned in Agisoft. The output is a txt file such as the IR_261121_ties_Gh.txt. 
Then in the Drift_correction_Randomforest.py file the IR_261121_ties_Gh.txt is read, and drift is modeled using random forest regression, finally the drift can be corrected in each photo pixel-wise based on the time in sec at which the photo was taken. 
