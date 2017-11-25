# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:43:16 2017

@author: chandler
"""
import sqlite3;
import numpy as np;
import cv2;
import os.path;
import shutil;

def generate_annotation():
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h,faceimages.image_id,facemetadata.sex,facemetadata.occluded"
    from_string = "faceimages, faces, facepose, facerect,facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    sqlite_path = '/home/tracking/work/src/aflw/data/aflw.sqlite';
#    sqlite_path = '/home/chandler/Desktop/hyperface/src/aflw.sqlite';

    
    list_path = '/home/tracking/work/src/frcnn/frcnn.txt';    
#    list_path = '/home/chandler/Desktop/frcnn/test.txt';   
    
    dataset = sqlite3.connect(sqlite_path);
    cursor = dataset.cursor();

#    gender_dict = {'m':1,'f':0};

    landmark_list = [];
    visibility_list = [];
    pose_list = [];
    
    list_file = open(list_path,'w');
    
    for row in (cursor.execute(query_string)):
#        if occluded, skip
        if (row[11]==1):
            continue;
        
        select_str = "coords.feature_id, coords.x, coords.y"
        from_str = "featurecoords coords"
        where_str = "coords.face_id = {}".format(row[1])
        query_str = "SELECT " + select_str + " FROM " + from_str + " WHERE " + where_str
    	
        landmark = np.zeros((21,2)).astype(np.float32);
        visibility = np.zeros((21,2)).astype(np.int32);
    
        tmp_cursor = dataset.cursor();			
        
        
        for xx in tmp_cursor.execute(query_str):
            landmark[xx[0]-1][0] = (2*xx[1]/float(row[7]))-1;#(xx[1] - center_x)/float(row[7])
            landmark[xx[0]-1][1] = (2*xx[2]/float(row[8]))-1;#(xx[2] - center_y)/float(row[8])
            visibility[xx[0]-1][0] = 1;
            visibility[xx[0]-1][1] = 1;
        
        landmark = landmark.reshape(-1,42);
        visibility = visibility.reshape(-1,42);
        
        tmp_cursor.close();
        
        pose = np.asarray([row[2],row[3],row[4]]).astype(np.float32);    
     
            
        landmark_list.append(landmark);
        visibility_list.append(visibility);
        pose_list.append(pose);
            
        list_file.write(row[0]+'\n');

            
    cursor.close();
    list_file.close();
            
        

        
if __name__ == "__main__":
    generate_annotation();