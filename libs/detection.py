# -*- coding: utf8 -*-
from math import ceil
import numpy as np
import cv2

from Yolo_tensorRT.TensorRT_yolo import BBox, TensorRT_Yolo
from libs.postprocess import Node, NodeBucket, carLP_postprocess

# dt = np.dtype([('name',  'U64'),('score', 'f4'),('x1', 'i4')])

def eng_to_thai(best_word):
    thai_dic = {
    'go_gai':'ก', 'ko_kai':'ข', 'ko_kuad':'ฃ', 'ko_kuai':'ค', 'kho_khon':'ฅ',
    'ko_rakang':'ฆ', 'ngo_ngoo':'ง', 'jo_jaan':'จ', 'cho_ching':'ฉ', 'cho_chaang':'ช',
    'so_so':'ซ', 'cho_gachar':'ฌ', 'yo_ying':'ญ', 'do_chadaa':'ฎ', 'do_badak':'ฏ',
    'to_taan':'ฐ', 'to_monto':'ฑ', 'to_puu_tao':'ฒ', 'no_nen':'ณ', 'do_dek':'ด',
    'do_dao':'ต', 'to_tung':'ถ', 'to_tahaan':'ท', 'to_tong':'ธ', 'no_noo':'น',
    'bo_baimai':'บ', 'bo_blaa':'ป', 'po_perng':'ผ', 'fo_faa':'ฝ', 'po_paan':'พ',
    'fo_fan':'ฟ', 'po_sam_pao':'ภ', 'mo_maa':'ม', 'yo_yak':'ย', 'ro_rer':'ร',
    'lo_ling':'ล', 'wo_wen':'ว', 'so_saa_laa':'ศ', 'so_rsii':'ษ', 'so_ser':'ส',
    'ho_hiib':'ห', 'lo_julaa':'ฬ', 'or_aang':'อ', 'ho_nokhoo':'ฮ'
    }
    if best_word in thai_dic.keys():
        # print(thai_dic[number])
        best_word = thai_dic[best_word]
    return best_word

def license_plate_recognition(inputData: np.ndarray, detector: TensorRT_Yolo, max_buckets: int=10):

    result_img = inputData.copy()
    detections = detector.predict(inputData)
    # print(detections)

    if max_buckets <= 0:
        max_buckets = 10
    buck_width = int(ceil(result_img.shape[1] / max_buckets))
    BucketMap = [ NodeBucket(i,buck_width) for i in range(max_buckets)]
    for wid, word in enumerate(detections):
        b1 = (word.xmin // buck_width)
        b2 = (word.xmax // buck_width) + 1
        for x in range(b1, b2):
            # new_score = (rr-ll) / bucket.width * word.score
            ll = max(word.xmin, BucketMap[x].left)
            rr = min(word.xmax, BucketMap[x].right)
            interSection = max(0, rr-ll)
            # new_score = interSection / buck_width * word.score
            new_score = interSection / word.width * word.score
            # print("{} new_score={} ll={}, rr={}, width={}, old_score={}".format(word.clsName, new_score,ll,rr, word.width, word.score))
            # print("{} new_score={} ll={}, rr={}, width={}, old_score={}".format(word.clsName, new_score,ll,rr, buck_width, word.score))
            
            if new_score:
                node = Node(
                    score=new_score,
                    word=word.clsName,
                    wordID=wid,
                    bucketID=x
                )
                BucketMap[x].add_nodes([node])

    LP_number = ""
    BuckPath = []

    # for buck in BucketMap:
    #     print(buck.nodeDict.items())

    # Basic CTC, pick each best node then merge repeat
    for Buck in BucketMap:
        if Buck.best.wordID != -1:
            # print(Buck.best.word)
            Buck.best.word = eng_to_thai(Buck.best.word)
            if len(BuckPath) == 0:
                BuckPath.append(Buck)
                LP_number += Buck.best.word
                # print('len = 0  Buck.best.word:',Buck.best.word)
            else:
                if BuckPath[-1].best.wordID == Buck.best.wordID:
                    BuckPath[-1].right = Buck.right
                    for _key, node in Buck.nodeDict.items():
                        BuckPath[-1].add_node(node)
                else:
                    BuckPath.append(Buck)
                    LP_number += Buck.best.word
                    # print('Buck.best.word:',Buck.best.word)
        # print('LP_number:',LP_number)
    x1 = 0
    for i in range(max_buckets):
        cv2.line(result_img,(x1, 0),(x1, result_img.shape[0]-1),(0,255,0),1)
        if (x1 + buck_width) >= result_img.shape[1]:
            x1= result_img.shape[1]-1
        else:
            x1 = x1 + buck_width
        cv2.line(result_img,(x1, 0),(x1, result_img.shape[0]-1),(0,0,255),1)
    ### draw bbox
    for bbox in detections:
        cv2.rectangle(result_img, bbox.pt1, bbox.pt2, (0, 0, 255), 1)
    
    return LP_number, BuckPath, result_img


def vehicle_detection(inputData: np.ndarray, detector: TensorRT_Yolo):

    result_img = inputData.copy()
    detections = detector.predict(inputData)

    ### draw bbox
    for bbox in detections:
        cv2.rectangle(result_img, bbox.pt1, bbox.pt2, (0, 0, 255), 3)
    
    return detections, result_img
    
