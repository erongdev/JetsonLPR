import os
import glob
import cv2

from pprint import pprint
# import pdb



def get_imgPaths(img_root):
    img_paths= glob.glob(os.path.join(img_root, '*.jpg'))
    return sorted(img_paths)

def batch_imgReader(img_paths, batch_size=1):
    
    batch_paths= []
    batch_images= []
    for img_path in img_paths:
        img= cv2.imread(img_path)
        batch_paths.append(img_path)
        batch_images.append(img)
        if len(batch_images)%batch_size==0:
            yield batch_paths, batch_images
            batch_paths.clear()
            batch_images.clear()

    
    if len(batch_images) >0:
        yield batch_paths, batch_images
        batch_paths.clear()
        batch_images.clear()


def batch_oneVideo(cap, batch_size=1):
    assert(cap.isOpened()), "Can't open video in batch_oneVideo function"

    batch_frames= []
    while True:
        ret, frame= cap.read()
        if ret==False:
            break
        
        batch_frames.append(frame)
        if len(batch_frames)%batch_size==0:
            yield batch_frames
            batch_frames.clear()
            

    if len(batch_frames)>0:
        yield batch_frames
    cap.release()
        

def main():
    img_root="data/lprSampleRenameJPG"
    img_paths= get_imgPaths(img_root)
    for batch_imgPaths, batch_imgs in batch_imgReader(img_paths, batch_size=2):
        for img_path, img in zip(batch_imgPaths, batch_imgs):
            print(img_path)
            cv2.imshow("Frame", img)
            cv2.waitKey(1)
    
    #---- batch_oneVideo
    video_path="test.mp4"
    cap= cv2.VideoCapture(video_path)
    total_frame_inFile= cap.get(cv2.CAP_PROP_FRAME_COUNT)    
    frame_idx= 0
    for batch_frames in batch_oneVideo(cap, batch_size=2):
        for frame in batch_frames:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            frame_idx+=1
            print("TotalFrame: {}, FrameIdx: {}".format(total_frame_inFile, frame_idx))
    
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()