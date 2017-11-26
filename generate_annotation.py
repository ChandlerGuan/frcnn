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
import random;
from lxml import etree, objectify
from lxml.etree import Element, SubElement, tostring

#def write_xml(directory,filename,width,height,x1,y1,x2,y2):
#    E = objectify.ElementMaker(annotate=False)
#    anno_tree = E.annotation(
#      E.folder('aflw'),
#      E.filename(os.path.basename(filename)),
#      E.size(
#        E.width(width),
#        E.height(height),
#        E.depth(3)
#      ),
#      E.object(
#        E.name('face'),
#        E.difficult(0),
#        E.bndbox(
#            E.xmin(x1),
#            E.ymin(y1),
#            E.xmix(x2),
#            E.ymax(y2)
#        )
#      )
#    )
##    print(os.path.join(directory,os.path.splitext(filename)[0]+".xml"))
#    etree.ElementTree(anno_tree).write(open(os.path.join(directory,os.path.splitext(filename)[0]+".xml"),'w'),
#                      pretty_print=True)

def write_xml(directory,filename,width,height,x1,y1,x2,y2):
    node_root = Element('annotation')
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(filename)
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'face'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(x1)
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(y1)
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(x2)
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(y2)
#    print(os.path.join(directory,os.path.splitext(filename)[0]+".xml"))
    etree.ElementTree(node_root).write(open(os.path.join(directory,os.path.splitext(filename)[0]+".xml"),'w'),
                      pretty_print=True)                      


def generate_annotation():
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h,faceimages.image_id,faceimages.width,faceimages.height,facemetadata.sex,facemetadata.occluded"
    from_string = "faceimages, faces, facepose, facerect,facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    sqlite_path = '/home/tracking/work/src/aflw/data/aflw.sqlite';
#    sqlite_path = '/home/chandler/Desktop/hyperface/src/aflw.sqlite';
    list_path = '/home/tracking/work/src/frcnn/frcnn.txt';    
#    list_path = '/home/chandler/Desktop/frcnn/test.txt';   
    
    segment_path = '/home/tracking/work/src/frcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/';    
    
    annotation_path = '/home/tracking/work/src/frcnn/data/VOCdevkit2007/VOC2007/Annotations/';
    
    dataset = sqlite3.connect(sqlite_path);
    cursor = dataset.cursor();

#    gender_dict = {'m':1,'f':0};

    landmark_list = [];
    visibility_list = [];
    pose_list = [];
    
    list_file = open(list_path,'w');
    segment_list = [];
    
    for row in (cursor.execute(query_string)):
#        if occluded, skip
        if (row[13]==1):
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
        
        img_width = int(row[10]);
        img_height = int(row[11]);
        bbox_x1 = int(row[5]);
        bbox_y1 = int(row[6]);
        bbox_x2 = int(row[5])+int(row[7]);
        bbox_y2 = int(row[6])+int(row[8]);
               
        write_xml(annotation_path,os.path.basename(row[0]),img_width,img_height,
                  bbox_x1,bbox_y1,bbox_x2,bbox_y2)
#        list_file.write(row[0]+' face '+str(float(row[5]))+' '+str(float(row[6]))
#                        +' '+str(float(row[5])+float(row[7]))+' '+str(float(row[6])+float(row[8]))+'\n');
        if (bbox_x1<=0 or bbox_y1<0 or bbox_x2>=img_width or bbox_y2>=img_height):
            print(row[0])
            continue;

        if (not os.path.splitext(os.path.basename(row[0]))[1]=='.png'):
            segment_list.append(os.path.splitext(os.path.basename(row[0]))[0]);
            
    cursor.close();
    list_file.close();
    
    print(len(segment_list))
    
    train_split = 0.9;
    train_split = int(len(segment_list)*train_split);
    random.shuffle(segment_list);
    train_file = open(os.path.join(segment_path,'trainval.txt'),'w');
    test_file = open(os.path.join(segment_path,'test.txt'),'w');
    for i in range(train_split):
        train_file.write(segment_list[i]+'\n');
    for i in range(train_split,len(segment_list)):
        test_file.write(segment_list[i]+'\n');
    train_file.close();
    test_file.close();
            
        

        
if __name__ == "__main__":
    generate_annotation();