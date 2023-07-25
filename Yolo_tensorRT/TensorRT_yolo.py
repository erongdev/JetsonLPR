import os
from Yolo_tensorRT.yolo.yolo_with_plugins import TrtYOLO
from Yolo_tensorRT.yolo.yolo_to_onnx import convert_yolo_to_onnx
from Yolo_tensorRT.yolo.onnx_to_tensorrt import build_engine
import pycuda

import pdb


def build_trt_engine(model, category_num, batch_size=1, do_int8=False, int8_calib_images= "calib_images"):
    """
    model:'[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'
              Ex: ./model/yolov3-416x416
    
    category_num: 'number of object categories [80]'
    save_trt_model_path: 建立完的tensorRT model 的儲存位置
    """
    #Model to onnx
    convert_yolo_to_onnx(model, batch_size= batch_size)
    
    #onnx to tensorRT engine
    engine= build_engine(model, do_int8= do_int8, \
                        dla_core=-1, batch_size= batch_size,verbose=False)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    save_trt_model_path= '{}.trt'.format(model)
    with open(save_trt_model_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % save_trt_model_path)
    return


def get_classCount_inCfg(cfg_file):
    with open(cfg_file)as f:
        for one_line in f.readlines():
            one_line=one_line.strip()
            if 'classes' in one_line:
                class_count= int(one_line.split('=')[-1])
                return class_count

def get_inputHWShape_inCfg(cfg_file):
    w,h=0,0
    with open(cfg_file)as f:
        for one_line in f.readlines():
            one_line= one_line.strip()
            if 'width' in one_line:
                w= int(one_line.split('=')[-1])
            if 'height' in one_line:
                h= int(one_line.split('=')[-1])
        
            if w >0 and h>0:
                return (h,w)

    


class BBox:
    def __init__(self, xmin, ymin, xmax, ymax, score, cls_name):
        
        self.xmin= xmin
        self.ymin= ymin
        self.xmax= xmax
        self.ymax= ymax


        self.height= ymax - ymin
        self.width=  xmax - xmin
        self.area=self.height*self.width
        
        self.pt1=(self.xmin, self.ymin)
        self.pt2=(self.xmax, self.ymax)
        self.clsName=cls_name
        self.score= score

    def __eq__(self, other):
        if self.xmin!=other.xmin or self.xmax!=other.xmax:
            return False
        if self.ymin!=other.ymin or self.ymax!=other.ymax:
            return False
        if self.clsName!=other.clsName:
            return False
        if self.score!=other.score:
            return False
        return True

    def __lt__(self, other):
        return self.score < other.score

    @property
    def lx(self):
        return self.xmin
    
    @property
    def rx(self):
        return self.xmax
    
    @property
    def ly(self):
        return self.ymin
    
    @property
    def ry(self):
        return self.ymax

    @property
    def word(self):
        return self.clsName
    
    @word.setter
    def word(self, cls_name):
        self.clsName= cls_name

    def __repr__(self):
        show_msg="[xmin, ymin, xmax, ymax, clsName, score]=[{}, {}, {}, {}, {}, {}]".format(\
            self.xmin, self.ymin, self.xmax, self.ymax, self.clsName, self.score)
        return show_msg
    
    def draw_bbox(self, img):
        pass


    def get_BBoxCorner(self):
        """
        Get leftTop(x,y), rightDown(x,y)
        """
        return [self.xmin, self.ymin, self.xmax, self.ymax]        

    def center_X(self):
        return self.center_XY()[0]
    def center_Y(self):
        return self.center_XY()[1]    
    def center_XY(self):
        return [int((self.xmin+self.xmax)/2.0), int((self.ymin+self.ymax)/2.0)]
    
    def compute_center_distance(self, input_point):
        """
        input_point和此BBox中心點進行距離計算
        """
        distance= GetDistance(self.center_XY(), input_bbox)
        return distance

    def convertBack(self, bbox, scale):
        x, y, w, h= bbox
        xmin = int(round(x - (w / 2))*scale[0])
        xmax = int(round(x + (w / 2))*scale[0])
        ymin = int(round(y - (h / 2))*scale[1])
        ymax = int(round(y + (h / 2))*scale[1])

        if xmin<0:
            xmin=0

        if ymin<0:
            ymin=0

        return xmin, ymin, xmax, ymax


class TensorRT_Yolo:
    def __init__(self, trt_engine_path, class_txt, batch_size=1, letter_box=False, do_int8=False, int8_calib_images= "calib_images"):
        """
        trt_engine_path: tensorRT 的engine file，請勿必將此檔案與yolov3.cfg, yolov3.weights放在同一個資料夾，且主要名稱設為相同
        若trt不存在則會從trt所在的資料夾下生成TensorRT的engine file
        Ex:
        plate_yolov3-416.cfg
        plate_yolov3-416.weights
        plate_yolov3-416.trt

        class_txt: 所有label的對應class名稱
        """
        self.batch_size= batch_size     
        self.letter_box = letter_box     
        self.do_int8= do_int8
        self.int8_calib_images= int8_calib_images
        self.detector= self.load_model(trt_engine_path)
        self.label_class_dict= self.load_class_labels(class_txt)
        
    def load_class_labels(self, class_txt):
        with open(class_txt, 'r')as f:
            classes= [one_line.strip() for one_line in f.readlines()]
        return dict(enumerate(classes))

    def load_model(self, trt_engine_path):
        model_name= os.path.splitext(trt_engine_path)[0]
        cfg_file= '{}.cfg'.format(model_name)
        class_count= get_classCount_inCfg(cfg_file)
        hw_shape= get_inputHWShape_inCfg(cfg_file)

        if not os.path.exists(trt_engine_path):                    
            build_trt_engine(model_name, class_count, batch_size= self.batch_size, do_int8= self.do_int8, int8_calib_images= self.int8_calib_images)
        
        onnx_file= '{}.onnx'.format(model_name)
        # if os.path.exists(onnx_file):
        #     os.remove(onnx_file)


        detector= TrtYOLO(trt_engine_path, category_num= class_count, batch_size=self.batch_size)
        return detector

    def batch_predict(self, img_list, conf_th= 0.25):
        batch_boxes, batch_scores, batch_classes= self.detector.batch_detect(img_list, conf_th= conf_th, letter_box=self.letter_box)

        batchImg_bboxes= []
        for boxes, scores, classes in zip(batch_boxes, batch_scores, batch_classes):
            bboxes= []
            for box, score, one_class in zip(boxes, scores, classes):            
                class_str= self.label_class_dict[one_class]#.upper()
                xmin, ymin, xmax, ymax= box
                one_bbox= BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score= score, cls_name=class_str)
                bboxes.append(one_bbox)
            batchImg_bboxes.append(bboxes)
        
        return batchImg_bboxes

    

    def predict(self, img, conf_th=0.25):
        boxes, scores, classes= self.detector.detect(img, conf_th= conf_th, letter_box=self.letter_box)
    
        bboxes= []
        for box, score, one_class in zip(boxes, scores, classes):            
            class_str= self.label_class_dict[one_class]#.upper()
            xmin, ymin, xmax, ymax= box
            one_bbox= BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, score= score, cls_name=class_str)
            bboxes.append(one_bbox)
        return bboxes
