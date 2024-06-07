# Drift-correction-of-LWIR
Correction of thermal drift induced by changing core temperature and NU (non uniformity) of uncooled thermal cameras. 

Tie point are first extracted using the agisoft_py_export_tie_points.py file, after photos are aligned in Agisoft. The output is a txt file such as the IR_261121_ties_Gh.txt. 
Then in the drift is corrected in two steps, first forward drift using the "Drift_correction_step1_forward_drift.py" script then the side drift using the "Drift_correction_step2_forward_side_drift.py" .
