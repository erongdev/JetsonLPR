import cv2
import time
import pdb
import platform

def is_x86PC():
    return platform.processor()=='x86_64'


def replace_gstStr_forPC(gst_str):
    gst_plugins= gst_str.split('!')
    
    new_plugins= []
    for plugin in gst_plugins:
        if 'nvv4l2decoder' in plugin:#x86上沒有max-perform等變數
            new_plugins.append('nvv4l2decoder')
        elif 'nvv4l2h264enc' in plugin:#x86上沒有max-perform等變數
            new_plugins.append('nvv4l2h264enc')
        elif 'nvvidconv' in plugin:#x86上使用nvvidconvert替代
            new_plugin= plugin.replace('nvvidconv', 'nvvideoconvert')
            new_plugins.append(new_plugin)
        elif 'nv3dsink' in plugin:#顯示部份Jetson: nv3dsink, x86: nveglglessink
            new_plugin= plugin.replace('nv3dsink', 'nveglglessink')
            new_plugins.append(new_plugin)
        else:
            new_plugins.append(plugin)

    
    new_plugins= [plugin for plugin in map(lambda x:x.strip(), new_plugins)]
    new_gst_str= ' ! '.join(new_plugins)    
    return new_gst_str


def create_cv2VideoCapture(video_path):
    cap= cv2.VideoCapture(video_path)
    return cap


def create_gstreamerVideoCapture(video_path):
    if video_path.startswith('rtsp'):
        gst_str_start= 'rtspsrc location={} latency=0 is-live=True ! rtph264depay'.format(video_path)
    else:
        gst_str_start= 'filesrc location={}'.format(video_path)

    gst_str= "{} ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true".format(gst_str_start)
    if is_x86PC():
        gst_str= replace_gstStr_forPC(gst_str)


    # gst_str= "{} ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink".format(gst_str_start)
    # gst_str= "rtspsrc location='rtsp://admin:11111111@192.168.1.200:554/ch01/1' latency=0 is-live=True ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! 'video/x-raw, format=(string)BGRx' ! videoconvert ! appsink".format(video_path)
    print("Gst: {}".format(gst_str))
    cap= cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    return cap

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    
    print(gst_str)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def create_gstreamerRTSPVideoWriter(width, height, host='127.0.0.1', port=5000, bitrate=4000000):
    """
    Use udpsink to output x/rtp streaming, and send to RTSPServer(host, port)
    """
    gst_str= ('appsrc ! videoconvert ! nvvidconv ! '
           'nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true bitrate={bitrate} preset-level=1 ! video/x-h264,stream-format=byte-stream ! h264parse ! rtph264pay ! udpsink host={host} port={port} sync=false async=0'.format(bitrate= bitrate, host=host, port=port))
    #  ! h264parse ! rtph264pay ! udpsink host=192.168.33.162 port=5000    
        
    if is_x86PC():        
        gst_str= replace_gstStr_forPC(gst_str)
    
    writer= cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (width, height))
    assert(writer.isOpened()), "Can't open gstreamer writer: "#.format(output_videoPath)
    return writer


def create_gstreamerUDPSrc_ShowImg(width, height, port):
    gst_str= ("udpsrc port={port} caps='application/x-rtp, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nv3dsink sync=0".format(port=port))

    if is_x86PC():
        gst_str= replace_gstStr_forPC(gst_str)
    
    writer= cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (width, height))
    assert(writer.isOpened()), "Can't open gstreamer writer: "#.format(output_videoPath)
    return writer


def create_gstreamerVideoWriter(output_videoPath, width, height,fps): #add fps 
    gst_str= ('appsrc ! videoconvert ! nvvidconv ! '
            'nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! '
            'filesink location={}').format(output_videoPath)
    
    if is_x86PC():
        gst_str= replace_gstStr_forPC(gst_str)

    writer= cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    assert(writer.isOpened()), "Can't open gstreamer writer: {}".format(output_videoPath)
    return writer

def create_gstreamerVideoWriter_showImg(width, height):
    gst_str= 'appsrc ! queue ! videoconvert ! nvvidconv ! nv3dsink sync=0'#nv3dsink -e ---> nv3dsink, and sync must be 0 if you use rtsp

    if is_x86PC():
        gst_str= replace_gstStr_forPC(gst_str)
    
    writer= cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (width,height))
    assert(writer.isOpened()), "Can't open gstreamer writer showImg"
    return writer


def create_cv2Writer(output_videoPath, width, height):
    fourcc= cv2.VideoWriter_fourcc(*'H264')
    writer= cv2.VideoWriter(output_videoPath, fourcc, 30, (width,height))
    return writer

def create_gstreamerVideoWriter_andShow(video_path, width, height):    
    gst_str= 'appsrc ! videoconvert ! nvvidconv ! tee name=branch1 ! queue ! nv3dsink sync=0 branch1. ! queue ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location={}'.format(video_path)
        
    if is_x86PC():
        gst_str= replace_gstStr_forPC(gst_str)

    writer= cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (width,height))
    assert(writer.isOpened()), "Can't open gstreamer writer: {}".format(video_path)
    return writer


def main():
    USE_SHOW_VIDEO_WRITER= False
    rtsp_src= "rtsp://root:80661707@192.168.33.169/axis-media/media.amp"
    cap= create_gstreamerVideoCapture(rtsp_src)
    ret, frame= cap.read()
    if ret==False:
        print("Can't get first frame")
        return
    
    output_path= "output_video.mp4"
    if USE_SHOW_VIDEO_WRITER:
        show_video_wirter= create_gstreamerVideoWriter_andShow(output_path, width= frame.shape[1], height=frame.shape[0])
    else:
        writer= create_gstreamerVideoWriter(output_path, width=frame.shape[1], height=frame.shape[0])
        show_writer= create_gstreamerVideoWriter_showImg(width= frame.shape[1], height=frame.shape[0])

    while True:
        ret, frame= cap.read()
        if ret==False:
            break

        if USE_SHOW_VIDEO_WRITER:
            show_video_wirter.write(frame)
        else:
            show_writer.write(frame)
            writer.write(frame)
    
    cap.release()

    if USE_SHOW_VIDEO_WRITER:
        show_video_wirter.release()
    else:
        show_writer.release()
        writer.release()


def rtspStream_main():
    import subprocess
    rtsp_src= "rtsp://root:80661707@192.168.33.169/axis-media/media.amp"
    cap= create_gstreamerVideoCapture(rtsp_src)
    ret, frame= cap.read()
    if ret==False:
        print("Can't get first frame")
        return

    rtsp_writer= create_gstreamerRTSPVideoWriter(width= frame.shape[1], height= frame.shape[0], host='127.0.0.1', port=5000)
    
    if is_x86PC():
        subprocess.Popen("gst-launch-1.0 udpsrc port=5000 caps='application/x-rtp, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! h264parse ! nvv4l2decoder ! nveglglessink", shell=True)
    else:
        subprocess.Popen("gst-launch-1.0 udpsrc port=5000 caps='application/x-rtp, encoding-name=(string)H264, payload=(int)96' ! rtph264depay ! h264parse ! nvv4l2decoder ! nv3dsink", shell=True)
    while True:
        ret, frame= cap.read()
        if ret==False:
            break

        rtsp_writer.write(frame)
        

    cap.release()
    rtsp_writer.release()
    


if __name__=='__main__':    
    # main()
    rtspStream_main()
    
