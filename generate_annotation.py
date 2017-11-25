# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:43:16 2017

@author: chandler
"""
import h5py;
import sqlite3;
import numpy as np;
import cv2;
import os.path;
import shutil;

def create_h5():
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h,faceimages.image_id,facemetadata.sex,facemetadata.occluded"
    from_string = "faceimages, faces, facepose, facerect,facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    sqlite_path = '/home/tracking/work/src/aflw/data/aflw.sqlite';
#    sqlite_path = '/home/chandler/Desktop/hyperface/src/aflw.sqlite';
    img_path = ('/home/tracking/work/src/aflw/data/flickr/');    
    
#    h5_path = '/home/chandler/Desktop/hyperface/h5_dataset';
    h5_path = '/home/tracking/work/src/hyperface/h5_dataset';
    h5_prefix = 'hyperface';
#    list_path = '/home/chandler/Desktop/hyperface/hyperface_dataset.txt';
    list_path = '/home/tracking/work/src/hyperface/hyperface_dataset.txt';
    
    if os.path.exists(h5_path):
        shutil.rmtree(h5_path);    
    if not os.path.exists(h5_path):
        os.makedirs(h5_path);
        
    
#    h5_capacity = 4000;
    h5_capacity = 3200;    
    
    dataset = sqlite3.connect(sqlite_path);
    cursor = dataset.cursor();

#    gender_dict = {'m':1,'f':0};

    capacity_cnt = 0;
    h5_cnt = 0;
    landmark_list = [];
    visibility_list = [];
    image_list = [];
    pose_list = [];
    
    list_file = open(list_path,'w');
    

    for row in (cursor.execute(query_string)):
        if (row[11]==1):
            continue;
        
        select_str = "coords.feature_id, coords.x, coords.y"
        from_str = "featurecoords coords"
        where_str = "coords.face_id = {}".format(row[1])
        query_str = "SELECT " + select_str + " FROM " + from_str + " WHERE " + where_str
    	
        landmark = np.zeros((21,2)).astype(np.float32);
        visibility = np.zeros((21,2)).astype(np.int32);
    
        tmp_cursor = dataset.cursor();			
        
        img_raw = cv2.imread(img_path+row[0][2:]);
        img_shape = img_raw.shape;
        
        for xx in tmp_cursor.execute(query_str):
            landmark[xx[0]-1][0] = (xx[1]/float(img_shape[1]))-0.5;#(xx[1] - center_x)/float(row[7])
            landmark[xx[0]-1][1] = (xx[2]/float(img_shape[0]))-0.5;#(xx[2] - center_y)/float(row[8])
            visibility[xx[0]-1][0] = 1;
            visibility[xx[0]-1][1] = 1;
        
        landmark = landmark.reshape(-1,42);
        visibility = visibility.reshape(-1,42);
        
              
        
        tmp_cursor.close();
    
    #    img_raw = (np.asarray(cv2.imread(img_path+row[0][2:])).astype(np.float32))/255.0;

        img_raw = ((np.asarray(cv2.resize(img_raw,(227,227),interpolation=cv2.INTER_CUBIC)).astype(np.float32))-128)/255.0;
        
#        print(img_raw.shape);
        pose = np.asarray([row[2],row[3],row[4]]).astype(np.float32);    
    
        print(img_path+row[0]); 
        
        if len(img_raw.shape) !=3:
            continue;#img_raw = color.gray2rgb(img_raw)
        if len(img_raw.shape) !=3 or img_raw.shape[2] != 3:
            continue;
            
        img_raw = np.swapaxes(img_raw,0,2);
            
        image_list.append(img_raw);
        landmark_list.append(landmark);
        visibility_list.append(visibility);
        pose_list.append(pose);
        
        
        capacity_cnt = capacity_cnt + 1;
        if (capacity_cnt==h5_capacity):
            h5_name = os.path.join(h5_path,h5_prefix+str(h5_cnt)+'.h5');
            print("saving to:"+h5_name);
            h5_file = h5py.File(h5_name);
            h5_file['image'] = (np.asarray(image_list)).astype(np.float32);
            h5_file['landmark'] = (np.asarray(landmark_list)).astype(np.float32);
            h5_file['visibility'] = (np.asarray(visibility_list)).astype(np.float32);
#            h5_file['pose'] = (np.asarray(pose_list)).astype(np.float32);
            
            list_file.write(h5_name+'\n');
            h5_cnt = h5_cnt + 1;
            
            capacity_cnt = 0;
            landmark_list = [];
            visibility_list = [];
            image_list = [];
            pose_list = [];

            h5_file.close();
            
            
            
    
    h5_name = os.path.join(h5_path,h5_prefix+str(h5_cnt)+'.h5');
    h5_file = h5py.File(h5_name);
    h5_file['image'] = (np.asarray(image_list)).astype(np.float32);
    h5_file['landmark'] = (np.asarray(landmark_list)).astype(np.float32);
    h5_file['visibility'] = (np.asarray(visibility_list)).astype(np.float32);
#    h5_file['pose'] = (np.asarray(pose_list)).astype(np.float32);    
    list_file.write(h5_name+'\n');
    h5_file.close();  
            
    cursor.close();
    list_file.close();
            
        

        
if __name__ == "__main__":
    create_h5();