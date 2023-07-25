# !/usr/bin/python
# coding:utf-8
import os
import cv2
import glob
import time
import traceback
from ipcamCapture import *
import numpy as np
from collections import Counter
from Yolo_tensorRT import TensorRT_Yolo
import pycuda.autoinit
import time
import csv
import imagezmq
import simplejpeg
import zmq
from shapely import geometry
import json
from libs.detection import  license_plate_recognition
from libs.postprocess import  motorLP_postprocess
# sys.path.append('./') # run pyinstaller need open let system know config path
from config import *

import pdb

os.environ['TZ'] = 'Asia/Bangkok'
time.tzset()


def createFolder(targetPath): #建立資料夾
    if not os.path.exists(targetPath): #如果沒有此路徑沒有該資料夾
        os.makedirs(targetPath) #在此路徑建立資料夾
        # print (targetPath +" , not exist so we create it")

def postErrorMessage(errorMessage):
    conf = get_config()
    localtime = time.localtime()
    time_ymd = time.strftime("%Y%m%d", localtime)
    createFolder(conf.logPath)
    logPath = conf.logPath+'/'+str(time_ymd)
    logPath=time.strftime(logPath,time.localtime())
    localTime=time.localtime()
    
    with open (logPath,'a') as f:       
        f.write(time.strftime("%Y/%-m/%-d 上午 %H:%M:%S\n",localTime))
        f.write(errorMessage)

def writeCSV(targetPath,filename,localtime,location,moto_plate_num,image_path,objectID,avgSpeed):
    createCSV(targetPath,filename)
    time_date = time.strftime("%Y/%m/%d", localtime)
    time_hms = time.strftime("%H:%M:%S", localtime)
    with open(targetPath+'/'+filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow([m,time_date,time_hms,location,moto_plate_num,image_path])
        writer.writerow([time_date,time_hms,location,moto_plate_num,avgSpeed,image_path])

def createCSV(targetPath,filename): #建立資料夾
    if not os.path.exists(targetPath+'/'+filename): #如果沒有此路徑沒有該資料夾
        with open(targetPath+'/'+filename, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            # writer.writerow(['TrackerID','日期','時間', '地點','車牌號', '照片位置'])
            writer.writerow(['日期', '時間', '地點', '車牌號', '速度', '照片位置'])
            csvfile.close()
def if_inPoly(polygon, Points):
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)

def yoloCarPlate(img,detector_plate):
    LP_number, BucketMap, result_img = license_plate_recognition(img, detector_plate)
    # cv2.imshow('plate',result_img)
    # cv2.waitKey(0)
    # print(LP_number)
    plateType = 0 #0:無規則 ,5:新規則
    try:
        new_LP_number = motorLP_postprocess(BucketMap, plateType)
    except:
        new_LP_number = LP_number
    return new_LP_number

def detecNumber(batch, detector_plate, img, bboxes, ft):
    conf = get_config()
    for bbox in bboxes:
        # print(if_inPoly(conf.plate_zone[batch],bbox.pt1))
        rec_zone = if_inPoly(conf.plate_zone[batch],bbox.pt1) #Recognition Zone limit
        # if bbox.clsName=='box' and bbox.xmax-bbox.xmin > conf.plate_width and bbox.ymin > conf.plate_zone_ymin[batch] and bbox.ymax < conf.plate_zone_ymax[batch]:
        if bbox.clsName=='box' and bbox.xmax-bbox.xmin > conf.plate_width and rec_zone:
            box_img = img[bbox.ymin:bbox.ymax,bbox.xmin:bbox.xmax]
            plate_num = yoloCarPlate(box_img,detector_plate)
            # plate_num = plate_num.upper()
            drawBbox(img,bbox)
            drawPlateNumber(img, bbox, plate_num,ft)
            return plate_num

class voting:
    def __init__(self, ft):
        conf = get_config()
        self.conf = conf
        self.ft = ft
        self.ballotBox = []
        self.totalVotes = 0
        self.uploadInterval = 0
        self.uploadTime = 0
        self.emptyFrameThresh = 0
        self.previousPlatenumber = None

    #Rewrite the license plate number according to the voting results
    def drawOutputImage(self, frame, number , font_scale= 3, color=(0,0,255), thickness= 2):
        cv2.rectangle(frame, (25, 50), (350, 125), (255, 255, 255), -1) 
        self.ft.putText(frame, number,org=(50,100),fontHeight=60,
                color=color,thickness=thickness,line_type=cv2.LINE_AA,bottomLeftOrigin=True)

    def update(self, plate_number):
        if plate_number != None:
            if self.emptyFrameThresh < self.conf.empty_Frame_Thresh:
                self.emptyFrameThresh += 1
            self.ballotBox.append(plate_number)
            self.totalVotes += 1
        else:
            if self.emptyFrameThresh > 0:
                self.emptyFrameThresh -= 1
            if  self.emptyFrameThresh == 0:
                self.emptyFrameThresh = self.conf.empty_Frame_Thresh
                self.ballotBox.clear()
    def uploadResult(self, batch, origin_img):
        img = origin_img.copy()
        if self.totalVotes > self.conf.plate_Frame_Thresh:
            if self.ballotBox != []:
                c = Counter(self.ballotBox)
                [(plate, _)] = c.most_common(1)
                #Thread
                # uploadThread =threading.Thread(target=socketUpload, args=(batch, plate, img))
                # uploadThread.start()
                # uploadThread.join()
                # print(threading.active_count())
                if self.previousPlatenumber != plate :
                    self.drawOutputImage(img,  plate)
                    self.socketUpload(batch, plate, img, origin_img)
                    print('plate:', plate)
                # else:
                #     print('same plate number continuously. cooling down time + ' + str(self.conf.upload_Interval) + ' second')
                if self.uploadInterval < 1:
                    self.previousPlatenumber = None
                self.previousPlatenumber = plate
                self.ballotBox.clear()
                c.clear()
            self.totalVotes = 0
            self.uploadInterval = self.conf.upload_Interval
            self.uploadTime = time.time()
        if self.uploadInterval > 0:
            timer = time.time() - self.uploadTime
            self.uploadTime = time.time()
            self.uploadInterval = self.uploadInterval - timer
    def socketUpload(self, batch, plate, img, origin_img):
        lpr_plate_string = plate
        lpr_timestamp = time.time()
        if self.conf.save_image == True: # 正式上線之後可以把存圖功能移到socket存檔失敗之後才存到本地端(except裡面)
            localtime = time.localtime()
            time_ymdhm = time.strftime("%Y%m%d%H%M", localtime)
            time_ymd = time.strftime("%Y%m%d", localtime)
            createFolder(self.conf.result_image_path+'/'+time_ymd+'/'+'camID_'+str(batch))
            imagename = self.conf.result_image_path+'/'+time_ymd+'/'+'camID_'+str(batch)+'/'+time_ymdhm+'_'+lpr_plate_string+'.jpg'
            print(imagename)
            ###---------這裡存乾淨圖片----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cv2.imwrite(imagename, img)
            createFolder(self.conf.result_image_path+'_clean'+'/'+time_ymd+'/'+'camID_'+str(batch))
            clean_imagename = self.conf.result_image_path+'_clean'+'/'+time_ymd+'/'+'camID_'+str(batch)+'/'+time_ymdhm+'_'+lpr_plate_string+'.jpg'
            # print(clean_imagename)
            cv2.imwrite(clean_imagename, origin_img)
            ###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        camera_name = self.conf.cam_name_list[batch]
        camera_ip = self.conf.cam_ip_list[batch]
        jpeg_quality = 95
        myDict = {
            'lpr_plate_string':str(lpr_plate_string),
            'lpr_timestamp':lpr_timestamp,
            'camera_name':str(camera_name),
            'camera_ip':str(camera_ip),
            }
        sender = imagezmq.ImageSender(connect_to=self.conf.socket_ip,REQ_REP=True)
        sender.zmq_socket.setsockopt(zmq.LINGER, 0) 
        sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 10000) # wait time 10 sec if not success time out.
        # sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 10000 )
        try:
            jsondata= json.dumps(myDict)
            msg = jsondata
            # Encode images to jpeg format by simplejpeg function to improve transfer efficiency
            jpg_buffer = simplejpeg.encode_jpeg(img, quality=jpeg_quality,
                                                colorspace='BGR')
            reply = sender.send_jpg(msg, jpg_buffer)
        except:
            errorMessage="SOCKET ERROR \n"+"Can't send data to "+self.conf.socket_ip+"\n\n"#get error messages #"\n"+traceback.print_exc()+
            postErrorMessage(errorMessage)#log function 
        finally:
            sender.close()

def drawPlateNumber(frame, bbox, number, ft, font_scale= 3, color=(0,0,255), thickness= 2):
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # text_pos= (bbox.xmin, bbox.ymin-10)
    # cv2.putText(frame, number, (50,100), font,font_scale, color, thickness)
    ft.putText(frame, number,org=(50,100),fontHeight=60,
                color=color,thickness=thickness,line_type=cv2.LINE_AA,bottomLeftOrigin=True)

def drawText(frame, draw_str, pos, font_scale= 0.5,color=(0,0,255), thickness= 2):
    font= cv2.FONT_HERSHEY_COMPLEX    
    cv2.putText(frame, draw_str,pos, font,font_scale, color, thickness)

def drawBbox(frame, bbox, color=(0,255,0), thickness=2):
    cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color= color, thickness= thickness)
    text_pos= (bbox.xmin, bbox.ymin-10)
    # drawText(frame, bbox.clsName, text_pos) 

def drawBboxes(frame, bboxes, color=(0,255,0), thickness=2):
    for bbox in bboxes:
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color= color, thickness= thickness)
        text_pos= (bbox.xmin, bbox.ymin-10)
        drawText(frame, bbox.clsName, text_pos) 


def getVideoPaths(img_root):
    conf = get_config()
    if conf.video_type == 'video':
        img_paths= glob.glob(os.path.join(img_root, '*.avi'))
        img_paths += glob.glob(os.path.join(img_root, '*.mp4'))
        img_paths += glob.glob(os.path.join(img_root, '*.mkv'))
    if  conf.video_type == 'webcam':
        img_paths = conf.cameralist
   
    return sorted(img_paths)


def prepare_Detector(batch_size):
    conf =get_config()
    engine_path = conf.weight_box_path
    class_txt = conf.names_box_path
    # 新版TensorRT用(TensorRT 8 onnx 1.9.0)
    detector_box= TensorRT_Yolo(engine_path, class_txt= class_txt, batch_size= batch_size, do_int8=False, int8_calib_images='./Yolo_tensorRT/yolo/calib_data')
    engine_path = conf.weight_plate_path
    class_txt = conf.names_plate_path
    detector_plate= TensorRT_Yolo(engine_path, class_txt= class_txt, batch_size= 1, do_int8=False, int8_calib_images='./Yolo_tensorRT/yolo/calib_data')
    return detector_box,detector_plate

class video_writer():
    def __init__(self,video_paths,output_root):
        self.output_root = output_root
        self.batch_paths= []
        self.ipcams=[]
        self.writers=[None] * len(video_paths)
        self.conf = get_config()
        for (i,(video_path)) in enumerate(video_paths):
            if self.conf.camera_reading_mode==1:
                ipcam=IpcamCapture(video_path)
                ipcam.start()
                time.sleep(1)
                if not ipcam.capture.isOpened():
                    errorMessage = "CAMERA ERROR \n Can't open video_path: {}\n\n".format(video_path)
                    postErrorMessage(errorMessage)#log function 
                # assert(ipcam.capture.isOpened()), "Can't open video_path: {}".format(video_path)
            elif self.conf.camera_reading_mode==0:
                ipcam=cv2.VideoCapture(video_path) 
                time.sleep(1)
                assert(ipcam.isOpened()), "Can't open video_path: {}".format(video_path)
            
            self.ipcams.append(ipcam)
            self.batch_paths.append(video_path)
            # self.set_writer(i)
    def register(self,):
        pass
    def set_writer(self,cam_id):
        conf = get_config()
        t = time.localtime()
        now_time = time.strftime("%Y%m%d_%H%M%S",t)
        output_path= os.path.join(self.output_root, 'cam'+str(cam_id)+'_'+now_time)
        # print(output_path)
        if conf.camera_reading_mode==1:
            self.writers[cam_id] = self.ipcams[cam_id].setWriter(output_path+'.avi')
        elif conf.camera_reading_mode==0:
            self.writers[cam_id] = setWriter(output_path+'.avi',self.ipcams[cam_id])

def batchVideoReader(set_video_writer, batch_size=1):
    conf = get_config()
    ipcams = set_video_writer.ipcams
    batch_images= []
    batch_ret =[]
    for i,ipcam in enumerate(ipcams):
        if conf.camera_reading_mode==1:
            ret, frame = ipcam.getframe()
        elif conf.camera_reading_mode==0:
            ret, frame = ipcam.read()
        if ret==False:
            recatch = threading.Thread(target=recapture, args=(conf, recaptureLock, set_video_writer, i))
            recatch.daemon=True
            recatch.start()
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # print(recaptureLock.locked())
        batch_images.append(frame)
        batch_ret.append(ret)
        if len(batch_images)%batch_size==0:
            yield  batch_images,batch_ret
            batch_images.clear()
            batch_ret.clear()
        
    if len(batch_images)>0:
        yield  batch_images,batch_ret
        batch_images.clear()
        batch_ret.clear()

    
def main():  
    conf =get_config()
    video_path=conf.video_root
    video_paths= getVideoPaths(video_path)
    batch_size= conf.batch_size
    output_root=conf.result_path
    os.makedirs(output_root, exist_ok=True)
    detector_box,detector_plate= prepare_Detector(batch_size)
    set_video_writer = video_writer(video_paths,output_root)
    #cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Demo", 960, 540)
    ft = cv2.freetype.createFreeType2()#為了印泰文
    ft.loadFontData(fontFileName='./font/THSarabunNewItalic.ttf', id=0)
    plateCounter = []
    for i in range(batch_size):
        plateCounter.append(voting(ft))
        if conf.save_video == True:
            set_video_writer.set_writer(i)

    while True:
        start_time= time.time()
        gpu_forward_total= 0
        for (i,(batch_images,batch_ret)) in enumerate(batchVideoReader(set_video_writer, batch_size)):
            s_t= time.time()
            batch_bboxes= detector_box.batch_predict(batch_images, conf_th=0.6)
            gpu_forward_total += (time.time() - s_t)
            #print("FPS: {}".format( 1.0 / (time.time() - s_t)))
            for (j,( origin_img,  ret,  bboxes)) in enumerate(zip( batch_images, batch_ret, batch_bboxes)):
                img = origin_img.copy()
                # drawBboxes(img, bboxes)
                plate_number = detecNumber(j, detector_plate, img, bboxes, ft)
                #voting
                plateCounter[j].update(plate_number)
                #upload plate number
                plateCounter[j].uploadResult(j, origin_img)
                
                #draw start detec line(green line)
                # cv2.line(img, (0,conf.plate_zone_ymin[j]), (1920,conf.plate_zone_ymin[j]), (0, 255, 0), 2)
                # cv2.line(img, (0,conf.plate_zone_ymax[j]), (1920,conf.plate_zone_ymax[j]), (0, 255, 0), 2)
                points =  np.array(conf.plate_zone[j], np.int32)
                cv2.polylines(img, pts=[points], isClosed=True, color=(0, 255, 0), thickness=4)
                #save video
                if conf.save_video == True:
                    set_video_writer.writers[j].write(img)
                
                # cv2.line(img, (0,conf.plate_zone_ymax), (1920,conf.plate_zone_ymax), (0, 255, 0), 2)    
                img = cv2.resize(img,(960, 540), interpolation=cv2.INTER_AREA)
                if j ==0:
                    imstack=img
                else: 
                    imstack = np.hstack((imstack,img))
            #cv2.imshow('Demo',imstack)
        #key= cv2.waitKey(1)
        #if key==27:
            #break
        
        full_program_time= time.time() - start_time
        # print("Full program time: {}, FPS: {}".format(full_program_time, 1.0 / (full_program_time / len(video_path))))        
        # print("GPU forward time: {}, FPS: {}".format(gpu_forward_total,  1.0 / (gpu_forward_total / len(video_path))))
    for ipcam in set_video_writer.ipcams:
        if conf.camera_reading_mode==1:
            ipcam.stop()
        elif conf.camera_reading_mode==0:
            ipcam.release()
    #cv2.destroyAllWindows()

if __name__=='__main__':
    # while True:
        try:
            main()
        except:
            traceback.print_exc()
            errorMessage='SYSTEM ERROR \n'+traceback.format_exc()+'\n\n'#get error messages
            postErrorMessage(errorMessage)#log function 
