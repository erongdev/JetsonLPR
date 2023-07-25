import cv2
import time
import threading
from video_stream import create_gstreamerVideoWriter
import queue
recaptureLock = threading.Lock()


class ThreadVideoWriter:
    def __init__(self, save_root, fourcc, FPS, videoSize):
        self.save_root, self.fourcc, self.FPS, self.videoSize=save_root, fourcc, FPS, videoSize
        # self.writer= cv2.VideoWriter(save_root, fourcc, FPS, videoSize)
        self.writer= None
        self.queue= queue.Queue()
        self.is_stop= False
        wrtr = threading.Thread(target=self.keep_run)
        wrtr.start()


    def keep_run(self):
        frame = None
        while self.is_stop==False:
            if self.queue.empty():
                time.sleep(0.01)
                continue
            if not self.queue.empty():
                frame= self.queue.get()
                if frame is not None :
                    if self.writer is None:
                        self.writer = cv2.VideoWriter(self.save_root, self.fourcc, self.FPS, self.videoSize)
                    self.writer.write(frame)
        if self.writer is not None:
            self.writer.release()

    def write(self, frame):
        self.queue.put(frame)
    
    def release(self):
        self.is_stop=True


class ThreadVideoWriter_MingYuan:
    def __init__(self, save_root, fourcc, FPS, videoSize):
        self.writer= cv2.VideoWriter(save_root, fourcc, FPS, videoSize)
        self.queue= queue.Queue()
        self.is_stop= False
        wrtr = threading.Thread(target=self.keep_run)
        wrtr.start()


    def keep_run(self):
        while self.is_stop==False:
            if self.queue.empty():
                time.sleep(0.01)
                continue
            frame= self.queue.get()
            self.writer.write(frame)
        
        self.writer.release()

    def write(self, frame):
        self.queue.put(frame)
    
    def release(self):
        self.is_stop=True

def setWriter(save_root,capture): #cv2存放影像相關設定
    
    videoSize = (int( capture.get(cv2.CAP_PROP_FRAME_WIDTH) ), int( capture.get(cv2.CAP_PROP_FRAME_HEIGHT) )  )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = capture.get(cv2.CAP_PROP_FPS)
    if FPS >30:
        FPS=30
    # fourcc =cv2.VideoWriter_fourcc(*'XVID')
    fourcc =cv2.VideoWriter_fourcc(*'H264')
    #fourcc =cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc =cv2.VideoWriter_fourcc(*'avc1')
    
    #Writer = create_gstreamerVideoWriter(save_root, width=width, height=height,fps=FPS) #for Xavier (need deepstream)
    Writer = cv2.VideoWriter(save_root, fourcc, FPS, videoSize)   
    #Writer = ThreadVideoWriter(save_root, fourcc, FPS, videoSize)
    return Writer


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class IpcamCapture:
    def __init__(self, URL):

        self.Frame = []
        self.status = False
        self.isstop = False

	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
               
        self.videoSize = (int( self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) ), int( self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) )  )
        self.width = int( self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int( self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = self.capture.get(cv2.CAP_PROP_FPS)
        if self.FPS >30:
            self.FPS=30     
        #self.fourcc =cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc =cv2.VideoWriter_fourcc(*'H264')
        #self.fourcc =cv2.VideoWriter_fourcc(*'mp4v')
        #self.fourcc =cv2.VideoWriter_fourcc(*'avc1')
    #cv2.writer相關設定
    def setWriter(self,save_root):
        # return create_gstreamerVideoWriter(save_root, width=self.width, height=self.height,fps=self.FPS) #for Xavier (need deepstream)
        return cv2.VideoWriter(save_root, self.fourcc, self.FPS, self.videoSize)
    
    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, args=()).start()
        self.daemon=True
    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.status,self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()

def recapture(conf,recaptureLock,set_video_writer,batch):
    '''
    conf is for check camera_reading_mode and cameralist
    recatchcamlock is thread lock
    set_video_writer is the class video_writer at detection_main.py
    '''
    # print('cam False')
    if recaptureLock.locked() == False:
        # print('lock',recatchcamlock.locked())
        recaptureLock.acquire(blocking=True, timeout=1)
        if conf.camera_reading_mode==1:
            set_video_writer.ipcams[batch].stop()
            time.sleep(10)
            set_video_writer.ipcams[batch] =IpcamCapture(conf.cameralist[batch])
            set_video_writer.ipcams[batch].start()
            time.sleep(3)
        if conf.camera_reading_mode==0:
            set_video_writer.ipcams[batch].release()
            time.sleep(10)
            set_video_writer.ipcams[batch] =cv2.VideoCapture(conf.cameralist[batch], cv2.CAP_FFMPEG)
            time.sleep(3)
        recaptureLock.release()


def main():
    url=''
    ipcam = IpcamCapture(url)
    ipcam.start()
    time.sleep(1)
    
    save_path='./ipCamRecording.avi'
    Writer=ipcam.setWriter(save_path)
    while True:
               
        ret, frame_read = ipcam.getframe()
        if ret==False:
            break

        cv2.imshow("ipCam", frame_read)
        key= cv2.waitKey(30)
        
        if key==27:
            break
    
    
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()
