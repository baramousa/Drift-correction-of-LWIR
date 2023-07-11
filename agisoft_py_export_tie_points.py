# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:27:47 2023
# A script that print tie points and their coordinates and tempertures in each thermal photo they appear in
# The code was adapted from Alexey Pasumansky, see https://www.agisoft.com/forum/index.php?topic=10730. 
#The code can be implemented in Agisoft after the thermal photos have been aligned successfully.
#export format:	
#[camera_label,	x_proj,	y_proj, temperature, tie point index, number of projections, projection error]


@author: Albara
"""



import Metashape, math

doc = Metashape.app.document
#The chunk of thermal photos
chunk = doc.chunk
M = chunk.transform.matrix
crs = chunk.crs
# get the tie points
point_cloud = chunk.tie_points
#(tie points their correspoding projections i.e the photos they appear in)
projections = point_cloud.projections 
#(valid tie points)
points = point_cloud.points 
npoints = len(points)
#(track or id of all points in the cloud valid and invalid)
tracks = point_cloud.tracks 
# path to the output file
path = Metashape.app.getSaveFileName("Specify export path and filename:", filter ="Text / CSV (*.txt *.csv);;All files (*.*)")
file = open(path, "wt")
print("Script started...")

point_ids = [-1] * len(point_cloud.tracks)
# get the id of the valid tie points
for point_id in range(0, npoints):
	point_ids[points[point_id].track_id] = point_id  
points_proj = {}
####
#this code add the valid tie points in a dictionary as keys, and corresponding cameras/photos as values for the keys
cameras_valid=dict()
for camera in chunk.cameras:
    for proj in projections[camera]:#loop through tie points in the camera
        track_id = proj.track_id# get the track id of the point in that camera 
        point_id = point_ids[track_id]# get the point id according to its track id 

        if point_id < 0:# pass if the poid id is -1, which mean not valid
            continue
        if not points[point_id].valid: #skipping invalid points
            continue
        else:
            if point_id not in cameras_valid.keys():
                cameras_valid[point_id] = [camera]
            else:
                cameras_valid[point_id].append(camera)

for camera in chunk.cameras:
	print(camera)

	if not camera.transform:
		 continue
	T = camera.transform.inv()# transform back from wgs84 to pixel coordinate?
	calib = camera.sensor.calibration
	

	for proj in projections[camera]:# loop through every point in the cameras
		track_id = proj.track_id# get the track_id of that point
		point_id = point_ids[track_id]# accordingly get the point id 

		if point_id < 0:# first check if the point is valid
			continue
		if not points[point_id].valid: #skipping invalid points# second check if the tie point is valid
			continue

		point_index = point_id
		dist = camera.error(points[point_index].coord, proj.coord).norm() ** 2 # get the projection error
		if point_index in points_proj.keys():
			temp=0
			x, y = proj.coord # the x and y coordinates in pixel
			for band in camera.planes:
				temp=(band.photo.image()[round(x),round(y)][0])/40-100 # get the temp in degrees
			points_proj[point_index] = (points_proj[point_index] + "\n" + camera.label + "\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(x, y,temp,point_index,len(cameras_valid[point_index]),dist))
		else:
			temp=0
			x, y = proj.coord
			for band in camera.planes:
				temp=(band.photo.image()[round(x),round(y)][0])/40-100
			points_proj[point_index] = ("\n" + camera.label + "\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(x, y,temp,point_index,len(cameras_valid[point_index]),dist))

# write the infos in the file	
for point_index in range(npoints):

	if not points[point_index].valid:
		continue

	coord = M * points[point_index].coord
	coord.size = 3
	if chunk.crs:
		#coord
		X, Y, Z = chunk.crs.project(coord)
	else:
		X, Y, Z = coord

	line = points_proj[point_index]
	
	file.write("{:s}\n".format(line))

file.close()					
print("Finished")