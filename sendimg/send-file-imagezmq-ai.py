import socket
import time
import cv2
import imagezmq
import traceback
import simplejpeg
import json
import zmq

lpr_plate_string = '123456'
lpr_timestamp = time.time()
camera_name = 'no camera name'
camera_ip = '192.168.gg.ggg'
image=cv2.imread(r"./3fight.jpg")
jpeg_quality = 95
myDict = {
    'lpr_plate_string':str(lpr_plate_string),
    'lpr_timestamp':lpr_timestamp,
    'camera_name':str(camera_name),
    'camera_ip':str(camera_ip),
    }
# 192.168.33.172 is the host ip address of the sender
sender = imagezmq.ImageSender(connect_to='tcp://172.17.22.102:5555',REQ_REP=True)
#sender.zmq_socket.setsockopt(zmq.RCVTIMEO,1000)
time.sleep(2.0)
try:
    jsondata= json.dumps(myDict)
    msg = jsondata
    # Encode images to jpeg format by simplejpeg function to improve transfer efficiency
    jpg_buffer = simplejpeg.encode_jpeg(image, quality=jpeg_quality,
                                        colorspace='BGR')
    reply = sender.send_jpg(msg, jpg_buffer)
    print(reply)
except:
    print(traceback.print_exc())

