# -*- coding: utf-8 -*-
"""
Created on 31 March 2020 

@author: eray
"""
from easydict import EasyDict as edict
from datetime import datetime,timedelta

def get_config():
   
    d = datetime.today()#+timedelta(days=-7)   
    conf = edict()
    conf.camera_reading_mode=1 # 0:normal, 1:threading
    conf.video_type='webcam'#video or webcam usbcam
    # conf.cameralist=['./data/video/gate2_04.mp4']
    # conf.cameralist=['rtsp://eray:eRay80661707@192.168.7.90/MediaInput/h264'] 
  

    
    #-------------only can modify thi place--------------------------------------------------------------------------------------------------


    # if you need change camera ip modify this list
    conf.cameralist=['rtsp://root:Admin123@172.17.22.11/axis-media/media.amp?videocodec=h264&resolution=1920x1080&fps=30','rtsp://root:Admin123@172.17.22.12/axis-media/media.amp?videocodec=h264&resolution=1920x1080&fps=30']
    # upload plate data setting
    # if you need change the data of camera ip from socket please change this list 
    conf.cam_ip_list = ['172.17.22.11','172.17.22.12']
    conf.cam_name_list = ['location_in','location_out']
    # if you need change socket ip please modify here
    conf.socket_ip = 'tcp://172.17.22.102:5555'
    #conf.socket_ip2 = 'tcp://172.17.22.102:5556'
    # conf.socket_ip = 'tcp://192.168.254.69:5555'
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    
    
    conf.url='' # 空值代表web cam  0

    conf.video_root="./data/video" #test video path
    conf.result_path='./output' #video output  path
    conf.result_image_path = './output_images' #images output  path

    conf.weight_box_path = "./data/model/box/20230303/yolov3-416.trt"
    conf.names_box_path   = "./data/model/box/20230303/voc.names"
    conf.weight_plate_path = "./data/model/number/20230303/yolov3-224x96.trt"
    conf.names_plate_path   = "./data/model/number/20230303/voc.names"
    conf.logPath='./log'

    conf.save_video = False
    conf.save_image = False
    conf.plate_width = 80
    conf.plate_high = 30
    conf.batch_size = 2 #一次要辨識多少畫面，改了記得去改model不然會壞掉
    conf.plate_zone = [[(400,100),(1300,100),(1300,300),(1500,300),(1500,500),(1900,580),(1600,1000),(200,1000)],[(300,100),(1200,100),(1200,400),(1450,400),(1450,580),(1750,600),(1600,1000),(100,1000)]]
    conf.plate_zone_ymin = [250, 450]
    conf.plate_zone_ymax = [950, 900]
    conf.plate_Frame_Thresh = 90 # 1 sec 30 frame, calculate 3 sec to decide plate number
    conf.upload_Interval = 20 #sec 
    conf.empty_Frame_Thresh = 10 #frame

    return conf
