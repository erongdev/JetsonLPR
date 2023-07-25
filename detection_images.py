# !/usr/bin/python
# coding:utf-8
import os
import cv2
import time
from cv2 import waitKey
from ipcamCapture import *
from Yolo_tensorRT import TensorRT_Yolo
import pycuda.autoinit
import time
from libs.detection import vehicle_detection, license_plate_recognition
from libs.postprocess import carLP_postprocess, motorLP_postprocess, test_BucketMap
# sys.path.append('./') # run pyinstaller need open let system know config path
from config import *
import pdb


def createFolder(targetPath): #建立資料夾
    if not os.path.exists(targetPath): #如果沒有此路徑沒有該資料夾
        os.makedirs(targetPath) #在此路徑建立資料夾
        # print (targetPath +" , not exist so we create it")

def postErrorMessage(errorMessage,logPath):
    logPath=time.strftime(logPath,time.localtime())
    localTime=time.localtime()
    
    with open (logPath,'a') as f:       
        f.write(time.strftime("%Y/%-m/%-d 上午 %H:%M:%S\n",localTime))
        f.write(errorMessage)

def yoloCarPlate(img,detector_plate):
    LP_number, BucketMap, result_img = license_plate_recognition(img, detector_plate)
    # cv2.imshow('plate',result_img)
    # cv2.waitKey(0)
    # print(LP_number)
    plateType = 0 #0:無規則 ,5:台灣通用新規則
    try:
        new_LP_number = motorLP_postprocess(BucketMap, plateType)
    except:
        new_LP_number = LP_number
    return new_LP_number,result_img

def detecNumber(detector_plate,img, bboxes, ft):
    conf = get_config()
    for bbox in bboxes:
        if bbox.clsName=='box':
            box_img = img[bbox.ymin:bbox.ymax,bbox.xmin:bbox.xmax]
            plate_num,result_img = yoloCarPlate(box_img,detector_plate)
            plate_num = plate_num#.upper()
            drawPlateNumber(img, bbox, plate_num, ft)
    return  result_img,plate_num

def drawPlateNumber(frame, bbox, number, ft, font_scale= 3,color=(0,0,255), thickness= 2):
    print(number)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # text_pos= (bbox.xmin, bbox.ymin-10)
    # cv2.putText(frame, number, (50,100), font,font_scale, color, thickness)
    ft.putText(frame, number,org=(50,100),fontHeight=60,
                color=color,thickness=thickness,line_type=cv2.LINE_AA,bottomLeftOrigin=True)

        
def drawText(frame, draw_str, pos, ft, font_scale= 0.5,color=(0,0,255), thickness= 2):
    font= cv2.FONT_HERSHEY_COMPLEX    
    cv2.putText(frame, draw_str,pos, font,font_scale, color, thickness)
    
def drawBboxes(frame, bboxes, ft, color=(0,255,0), thickness=2):
    for bbox in bboxes:
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), color= color, thickness= thickness)
        text_pos= (bbox.xmin, bbox.ymin-10)
        drawText(frame, bbox.clsName, text_pos, ft) 

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


    
def main():  
    ft = cv2.freetype.createFreeType2()#為了印泰文
    ft.loadFontData(fontFileName='./font/THSarabunNewItalic.ttf', id=0)
    image_path='./test_image/032.png'
    img = cv2.imread(image_path)
    detector_box,detector_plate= prepare_Detector(1)
    bboxes= detector_box.predict(img, conf_th=0.25)
    drawBboxes(img, bboxes, ft)
    print(bboxes)
    result_img,plate_num = detecNumber(detector_plate,img, bboxes, ft)
    
    # plate_num,result_img = yoloCarPlate(img,detector_plate)
    # print(plate_num)
    cv2.imshow('Demo',img)
    key= cv2.waitKey(0)
    cv2.imwrite('./demo2.jpg',img)
if __name__=='__main__':
    main()

'''
{
'ก':'go_gai', 'ข':'ko_kai', 'ฃ':'ko_kuad', 'ค':'ko_kuai', 'ฅ':'kho_khon',
'ฆ':'ko_rakang', 'ง':'ngo_ngoo', 'จ':'jo_jaan', 'ฉ':'cho_ching', 'ช':'cho_chaang',
'ซ':'so_so', 'ฌ':'cho_gachar', 'ญ':'yo_ying', 'ฎ':'do_chadaa', 'ฏ':'do_badak',
'ฐ ':'to_taan', 'ฑ':'to_monto', 'ฒ':'to_puu_tao', 'ณ':'no_nen', 'ด':'do_dek',
'ต':'do_dao', 'ถ':'to_tung', 'ท':'to_tahaan', 'ธ':'to_tong', 'น':'no_noo',
'บ':'bo_baimai', 'ป':'bo_blaa', 'ผ':'po_perng', 'ฝ':'fo_faa', 'พ':'po_paan',
'ฟ':'fo_fan', 'ภ':'po_sam_pao', 'ม':'mo_maa', 'ย':'yo_yak', 'ร':'ro_rer',
'ล':'lo_ling', 'ว':'wo_wen', 'ศ':'so_saa_laa', 'ษ':'so_rsii', 'ส ':'so_ser',
'ห':'ho_hiib', 'ฬ':'lo_julaa', 'อ':'or_aang', 'ฮ ':'ho_nokhoo'
}

{
'go_gai':'ก', 'ko_kai':'ข', 'ko_kuad':'ฃ', 'ko_kuai':'ค', 'kho_khon':'ฅ',
'ko_rakang':'ฆ', 'ngo_ngoo':'ง', 'jo_jaan':'จ', 'cho_ching':'ฉ', 'cho_chaang':'ช',
'so_so':'ซ', 'cho_gachar':'ฌ', 'yo_ying':'ญ', 'do_chadaa':'ฎ', 'do_badak':'ฏ',
'to_taan':'ฐ ', 'to_monto':'ฑ', 'to_puu_tao':'ฒ', 'no_nen':'ณ', 'do_dek':'ด',
'do_dao':'ต', 'to_tung':'ถ', 'to_tahaan':'ท', 'to_tong':'ธ', 'no_noo':'น',
'bo_baimai':'บ', 'bo_blaa':'ป', 'po_perng':'ผ', 'fo_faa':'ฝ', 'po_paan':'พ',
'fo_fan':'ฟ', 'po_sam_pao':'ภ', 'mo_maa':'ม', 'yo_yak':'ย', 'ro_rer':'ร',
'lo_ling':'ล', 'wo_wen':'ว', 'so_saa_laa':'ศ', 'so_rsii':'ษ', 'so_ser':'ส ',
'ho_hiib':'ห', 'lo_julaa':'ฬ', 'or_aang':'อ', 'ho_nokhoo':'ฮ '
}
'''