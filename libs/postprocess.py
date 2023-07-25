from curses.ascii import isalpha
from typing import Dict, List
from pydantic import BaseModel
from math import ceil
import itertools
from Yolo_tensorRT.TensorRT_yolo import BBox

class Node(BaseModel):
    score: float=0.20
    word: str=""
    wordID: int=-1
    bucketID: int=-1
    
class NodeBucket:
    def __init__(self, bID: int, bWidth: int) -> None:
        self.id = bID
        self.left = bWidth*bID
        self.right = self.left + bWidth
        self.nodeDict = {"-1": Node()}
        self.best = self.nodeDict['-1']
    
    def add_node(self, nodeData: Node):
        curID = nodeData.wordID
        if curID != -1:
            if curID in self.nodeDict:
                if nodeData.score > self.nodeDict[curID].score:
                    self.nodeDict[curID].score = nodeData.score
                # self.nodeDict[curID].score += nodeData.score
                
            else:
                self.nodeDict[curID] = nodeData
            # update the best node
            if self.nodeDict[curID].score > self.best.score:
                self.best = self.nodeDict[curID]
    
    def add_nodes(self, nodeDatas: List[Node]):
        for nodeData in nodeDatas:
            curID = nodeData.wordID
            if curID != -1:
                if curID in self.nodeDict:
                    # self.nodeDict[curID].score += nodeData.score
                    pass
                else:
                    self.nodeDict[curID] = nodeData
                # update the best node
                if self.nodeDict[curID].score > self.best.score:
                    self.best = self.nodeDict[curID]

def confuse_num2eng(word: str):
    res = False
    check_dict= {
        '0':'D',
        '1':'I',
        '2':'Z',
        '3':'S',
        '5':'S',
        '6':'G',
        '7':'T',
        '8':'B'
    }
    if word in check_dict:
        res = True
        word = check_dict[word]
    return res, word

def find_candidate_num(buck: NodeBucket):
    res = False
    candidate = buck.best.word
    check_dict= {
        'D':'0', 'Q':'0',
        'I':'1',
        'Z':'2',
        'S':'5',
        'G':'6',
        'J':'7', 'T':'7',
        'B':'8'
    }
    if buck.best.word.isnumeric():
        res = True
        candidate = buck.best.word
    elif buck.best.word in check_dict:
        res = True
        candidate = check_dict[buck.best.word]
    else:
        thresh = 0.0
        for _key, val in buck.nodeDict.items():
            if val.word.isnumeric() and val.score > thresh:
                res = True
                thresh = val.score
                candidate = val.word

    return res, candidate

def find_candidate_eng(buck: NodeBucket):
    res = False
    candidate = buck.best.word
    check_dict= {
        '0':'D',
        '1':'I',
        '2':'Z',
        '3':'S',
        '5':'S',
        '6':'G',
        '7':'T',
        '8':'B',
    }
    if buck.best.word.isalpha():
        res = True
        candidate = buck.best.word
    elif buck.best.word in check_dict:
        res = True
        candidate = check_dict[buck.best.word]
    else:
        thresh = 0.0
        for _key, val in buck.nodeDict.items():
            if val.word.isalpha() and val.score > thresh:
                res = True
                thresh = val.score
                candidate = val.word

    return res, candidate


def motorLP_postprocess(BucketMap: List[NodeBucket], plateType: int):
    bucket_len = len(BucketMap)
    LP_number = ""

    progression_score = [ 0.0 for i in range(bucket_len)]
    for idx, Buck in enumerate(BucketMap):
        progression_score[idx] = progression_score[idx-1] + Buck.best.score
    # print(progression_score)

    lp_score = 0.0
    lp_start = 0
    lp_end = 0
    lp_len = 0

    if plateType == 0:
        for i in BucketMap:
            LP_number += i.best.word
            # print('BucketMap:',i.nodeDict.items())

    elif plateType == 1:
        # White base Black word 白底黑字 白牌
        # Motorcycle 機車
        
        ### do correct, only when word length is enough
        if len(LP_number) >= 6:
            ### check first 3 word
            pass
    
    elif plateType == 2:
        # 綠牌 機車
        pass
    elif plateType == 3:
        # 黃牌 機車
        pass
    elif plateType == 4:
        # Red base White word 紅底白字 紅牌
        # Motorcycle 機車
        if bucket_len < 4:
            for i in BucketMap:
                LP_number += i.best.word
        elif bucket_len < 5:
            # 2-2 AA-11
            lp_len = 4
            for i in range(lp_len-1, bucket_len):
                score = 0.0
                if i-lp_len < 0:
                    score = progression_score[i]
                else:
                    score = progression_score[i] - progression_score[i-lp_len]
                
                if score > lp_score:
                    lp_score = score
                    lp_start = i-lp_len+1
                    lp_end = i
            # print(lp_start, lp_end, lp_start+lp_len-1)

            for i in range(lp_start, lp_start+2):
                hasEng, eng = find_candidate_eng(BucketMap[i])
                LP_number += eng
            for i in range(lp_start+2, lp_start+lp_len):
                hasNum, num = find_candidate_num(BucketMap[i])
                LP_number += num
        else:
            ### bucket_len >= 6
            lp_len = 6 # default lp length
            for i in range(lp_len-1, bucket_len):
                score = 0.0
                if i-lp_len < 0:
                    score = progression_score[i]
                else:
                    score = progression_score[i] - progression_score[i-lp_len]
                
                if score > lp_score:
                    lp_score = score
                    lp_start = i-lp_len+1
                    lp_end = i
            # print(lp_start, lp_end, lp_start+lp_len-1)

            word_pattern = ""
            for i in range(3):
                if BucketMap[lp_start+i].best.word.isalpha():
                    word_pattern+="A"
                else:
                    word_pattern+="1"
            if word_pattern == "AAA":
                lp_start
                if lp_start+lp_len < bucket_len:
                    # AAA-1111 > 3-4
                    lp_len = 7
                # else:
                #     # AAA-111 > 3-3
                #     lp_len = 6
                for i in range(lp_start, lp_start+3):
                    hasEng, eng = find_candidate_eng(BucketMap[i])
                    LP_number += eng
                for i in range(lp_start+3, lp_start+lp_len):
                    hasNum, num = find_candidate_num(BucketMap[i])
                    LP_number += num
            elif word_pattern == "AA1":
                if lp_start > 0:
                    # AAA-1111 > 3-4
                    lp_start -= 1
                    lp_len = 7
                # else:
                #     # AAA-111 > 3-3
                #     lp_len = 6
                for i in range(lp_start, lp_start+3):
                    hasEng, eng = find_candidate_eng(BucketMap[i])
                    LP_number += eng
                for i in range(lp_start+3, lp_start+lp_len):
                    hasNum, num = find_candidate_num(BucketMap[i])
                    LP_number += num
            else:
                ### no rule
                for i in BucketMap:
                    LP_number += i.best.word
    elif plateType == 5:
        # for buck in BucketMap:
        #     print('TYPE 5 :',buck.nodeDict.items())
        #新的通用規則(因為只框了車牌沒有框是紅牌黃牌或白牌)
        for i in BucketMap:
            LP_number += i.best.word
        # print('LP_number:',LP_number)
        #目前台灣政府規定的所有機車車牌英數字排列可能性
        ruleList = [
                            'AAA1111',
                            'AAA111',
                            'AA1111',
                            'A11111',
                            '1AA111',
                            '11A111',
                            'A1A111',
                            '111AAA',
                            'AA111',
                            'AA11'
                            ]
        #會混淆的英數字
        confuseDict = {'D':'0', 'Q':'0',
                                        'I':'1',
                                        'Z':'2',
                                        'S':'5',
                                        'G':'6',
                                        'J':'7', 'T':'7',
                                        'B':'8',
                                        '0':'D',
                                        '1':'I',
                                        '2':'Z',
                                        '5':'S',
                                        '6':'G',
                                        '8':'B',
                                        '7':'T'
                                    } 
        word_pattern = ""
        changeList = []
        #第一次確定英數字的排列組合
        for i in range(len(LP_number)):
            if BucketMap[lp_start+i].best.word.isalpha():
                word_pattern+="A"
            else:
                word_pattern+="1"
        # print('word_pattern:',word_pattern)
        
        #如果排列組合不在規定的範圍內，就去找出可以變成正確規則的更換位置
        if word_pattern not in ruleList:
            for rule in ruleList:
                #字串長度跟規則不一樣直接放棄
                if len(rule) != len(LP_number):
                    continue
                diffPlace = []
                #把規則不一樣的位置紀錄下來
                for i in range(len(LP_number)):
                    if rule[i] != word_pattern[i]:
                        diffPlace.append(i)
                changeList.append(diffPlace)
            #先把需要更改的排列組合由少到多排序
            changeList.sort(key=lambda i : len(i),reverse=False)
            # print(changeList)
            if changeList != []:
                for change in changeList:
                    #若要改兩個以上的位置就乾脆放棄
                    if len(change) > 2:
                        continue
                    #更改範圍少於等於2才嘗試修正
                    else:
                        new_LP_number = ''
                        for letter in change:
                            if LP_number[letter] in confuseDict.keys():
                                new_LP_number = LP_number[:letter]+confuseDict[LP_number[letter]]+LP_number[letter+1:]
                            #一個修改組合裡面只要有一個位置不是混淆字就直接放棄這個組合
                            else:
                                break
                        # print(new_LP_number)
                        #只要一找到就馬上當正確解答
                        if new_LP_number != '':
                            # print(LP_number,'->',new_LP_number)
                            LP_number = new_LP_number
                            break

    return LP_number
    
def carLP_postprocess(BucketMap: List[NodeBucket], plateType: int):
    bucket_len = len(BucketMap)
    LP_number = ""

    progression_score = [ 0.0 for i in range(bucket_len)]
    for idx, Buck in enumerate(BucketMap):
        progression_score[idx] = progression_score[idx-1] + Buck.best.score
    print(progression_score)

    lp_score = 0.0
    lp_start = 0
    lp_end = 0
    lp_len = 0

    if plateType == 1:
        # White base Black word 白底黑字
        # Private sedan, Private light truck 自用小客(貨)車
        if bucket_len < 6:
            for i in BucketMap:
                LP_number += i.best.word
        else:
            lp_len = 6 # default lp length
            for i in range(lp_len-1, bucket_len):
                score = 0.0
                if i-lp_len < 0:
                    score = progression_score[i]
                else:
                    score = progression_score[i] - progression_score[i-lp_len]
                
                if score > lp_score:
                    lp_score = score
                    lp_start = i-lp_len+1
                    lp_end = i
            print(lp_start, lp_end, lp_start+lp_len-1)

            word_pattern = ""
            for i in range(2):
                if BucketMap[lp_start+i].best.word.isalpha():
                    word_pattern+="A"
                else:
                    word_pattern+="1"
            
            if word_pattern == "AA":
                if lp_start-1 >= 0:
                    if BucketMap[lp_start-1].best.word.isalpha():
                        # A-AA > AAA >3-4
                        lp_start -= 1
                        lp_len = 7

                        for i in range(lp_start, lp_start+3):
                            hasEng, eng = find_candidate_eng(BucketMap[i])
                            LP_number += eng
                        for i in range(lp_start+3, lp_start+lp_len):
                            hasNum, num = find_candidate_num(BucketMap[i])
                            LP_number += num
                    else:
                        # 1-AA > AA > 3-4 or 2-4
                        if BucketMap[lp_start+2].best.word.isalpha():
                            # AA[A] > 3-4
                            if lp_start+lp_len < bucket_len:
                                lp_len = 7
                            
                            for i in range(lp_start, lp_start+3):
                                hasEng, eng = find_candidate_eng(BucketMap[i])
                                LP_number += eng
                            for i in range(lp_start+3, lp_start+lp_len):
                                hasNum, num = find_candidate_num(BucketMap[i])
                                LP_number += num
                        else:
                            # AA[1] > 2-4
                            for i in range(lp_start, lp_start+2):
                                hasEng, eng = find_candidate_eng(BucketMap[i])
                                LP_number += eng
                            for i in range(lp_start+2, lp_start+lp_len):
                                hasNum, num = find_candidate_num(BucketMap[i])
                                LP_number += num
                else:
                    if BucketMap[lp_start+2].best.word.isalpha():
                        # AA[A] > 3-4
                        if lp_start+lp_len < bucket_len:
                            lp_len = 7
                        
                        for i in range(lp_start, lp_start+3):
                            hasEng, eng = find_candidate_eng(BucketMap[i])
                            LP_number += eng
                        for i in range(lp_start+3, lp_start+lp_len):
                            hasNum, num = find_candidate_num(BucketMap[i])
                            LP_number += num
                    else:
                        # AA[1] > 2-4
                        for i in range(lp_start, lp_start+2):
                            hasEng, eng = find_candidate_eng(BucketMap[i])
                            LP_number += eng
                        for i in range(lp_start+2, lp_start+lp_len):
                            hasNum, num = find_candidate_num(BucketMap[i])
                            LP_number += num

            elif word_pattern == "A1" or word_pattern == "1A":
                # A1 or 1A > 2-4
                for i in range(lp_start, lp_start+2):
                    LP_number += BucketMap[i].best.word
                for i in range(lp_start+2, lp_start+lp_len):
                    hasNum, num = find_candidate_num(BucketMap[i])
                    LP_number += num

            elif word_pattern == "11":
                # 11 > 4-2 or 111111
                for i in range(lp_start, lp_start+4):
                    hasNum, num = find_candidate_num(BucketMap[i])
                    LP_number += num
                for i in range(lp_start+4, lp_start+lp_len):
                    LP_number += BucketMap[i].best.word
            else:
                # no rule
                for i in BucketMap:
                    LP_number += i.best.word
                
    elif plateType == 2:
        # Gree base White word 綠底白字
        # Operating bus, Operating heavy truck 營業大客(貨)車
        pass
    elif plateType == 3:
        # White base Green word 白底綠字
        # Private bus, Private heavy truck 自用大客(貨)車
        pass
    elif plateType == 4:
        # White base Red word 白底紅字
        # Operating sedan 營業小客車
        pass
    elif plateType == 5:
        # Green base White word 綠底白字
        # Operating trailer 營業拖車
        pass
    elif plateType == 6:
        # White base Green word 白底綠字
        # Private trailer 自用拖車
        pass
    
    return LP_number


def test_BucketMap(max_steps: int, lp_width: int, wordDetection: List[BBox]):
    if max_steps <= 0:
        max_steps = 10
    stepW = int(ceil(lp_width / max_steps))
    BucketMap = [ NodeBucket(i,stepW) for i in range(max_steps)]
    for wid, word in enumerate(wordDetection):
        b1 = (word.xmin // stepW)
        b2 = (word.xmax // stepW) + 1
        for x in range(b1, b2):
            ### calculate word's new score in current slice
            ll = max(word.xmin, x*stepW)
            rr = min(word.xmax, (x+1)*stepW)
            interSection = max(0, rr-ll)
            slice_ratio = interSection / word.width
            if slice_ratio:
                node = Node(
                    score=slice_ratio*word.score,
                    word=word.clsName,
                    wordID=wid,
                    bucketID=x
                )
                BucketMap[x].add_nodes([node])

    LP_number = ""
    BuckPath = []

    for buck in BucketMap:
        print(buck.left, buck.right, buck.nodeDict.items())

    # Basic CTC, pick each best node then merge repeat
    for Buck in BucketMap:
        if Buck.best.wordID != -1:
            if len(BuckPath) == 0:
                BuckPath.append(Buck)
                LP_number += Buck.best.word
                print("first add")
            else:
                if BuckPath[-1].best.wordID == Buck.best.wordID:
                    BuckPath[-1].right = Buck.right
                    for _key, node in Buck.nodeDict.items():
                        BuckPath[-1].add_node(node)
                    print("same")
                else:
                    BuckPath.append(Buck)
                    LP_number += Buck.best.word
                    print("add")
        else:
            print("pass")
    
    for buck in BucketMap:
        print(buck.left, buck.right, buck.nodeDict)

    print('\nLP_number={}\n'.format(LP_number))
    for buck in BuckPath:
        print(buck.left, buck.right, buck.best, buck.nodeDict)

