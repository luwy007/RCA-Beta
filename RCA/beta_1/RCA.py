# -*- coding=utf-8 -*-
'''
Created on 2016年2月22日

@author: YANG
'''
'''
DATA INFO 
0~3：时间戳、小区名等无效信息

  9: 小区下行速率
 17：RRC建立成功率
 25：E-RAB建立成功率
 26：E-RAB掉话率
 27：同频切换成功率
 18,19,23,37,45,51,56经过之前的检测，都是完全无变化项，即给分类带来零帮助
 28,29,30有大量的NIL数据

'''

from Beta_1.RCA_Predict import RCA_Predict
from Beta_1.RCA_Train import RCA_Train
import time


def RCA(IGLIMIT=0.1, ratio=3):
    
    #===========================================================================
    # 参数获取，包括 path, fileName, ratio, IGLIMIT, label
    # path 训练数据的文件路径
    # fileName 训练数据文件名
    # ratio 模型的正负样本比例
    # IGLIMIT information gain的下限，用于控制决策树的生长。越小，决策树越复杂，得到的根因个数越多，可能包含的随机噪声越大
    # label 需要分析的KPI, 用几个简写代替
    # {Download Average Rate: 1, 
    #  RRC Setup Success Rate:2, 
    #  ERAB Setup Success Rate:3, 
    #  ERAB Call Drop Rate:4, 
    #  Intra-Frequency Handover Success Rate:5}
    #=========================================================================== 
    path = "tempdata"
    fileName = "1"
    labelIndex = 2
    labelDic = [9,17,25,26,27]
    filteredFea = []
    #print("whether use the default settings(y/n):")
    #===========================================================================
    # 这边的参数输入逻辑也需要修改，修改成从文件中读取参数列表     2/23
    #===========================================================================
    if(False and input()=='y'):
        print("please input the path of the data:")
        path = input()
        print("please input the file name of the data:")
        fileName = input()
        print("labelIndex:")
        labelIndex = input()
        print("IGLIMIT:")
        IGLIMIT = input()
        print("ratio:")
        ratio = input()
        print("filteredFea:")
        while(True):
            try:
                receiver = int(input())
                filteredFea.append(receiver)
            except:
                break
        
    trainer = RCA_Train(path, fileName, labelDic[labelIndex-1], IGLIMIT, ratio)
    predictor = RCA_Predict(path, fileName, labelDic[labelIndex-1])

    #showTime()
    #print("training is running\n......")
    trainer.Train(filteredFea)  
    return predictor.Predict(filteredFea)
    
def showTime():
    print(time.strftime("%Y-%m-%d  %H:%M:%S",time.localtime()))

def intersection(prediction1, prediction2):
    #===========================================================================
    # 用来衡量prediction1(dic类型)与prediction2之间的相近程度
    # 现有策略：
    # 每个prediction最多采用前十个元素，求交集，并给出各个元素的熵增益之和，用以进一步判断模型训练得出结果的稳定性
    # 至少有三个相同的才算stable
    #===========================================================================
    result = {}
    for item in prediction1:
        if(prediction2.__contains__(item)):
            result[item] = prediction1[item]+prediction2[item]
    
    return result

def main():
    #===========================================================================
    # 此处的处理逻辑存在较大的安全隐患，即对训练结果出现极端情况时丧失处理能力
    # 极端情况指： 当训练参数调至ratio很大，IGLIMIT很小，决策树既无法长大，root cause亦应该很稳定时，
    # 依旧不能满足 5组数据有3个共同的情况！
    #===========================================================================
    IGLIMIT = 0.07
    ratio = 2
    Found = False
    prediction1 = {}
    while(not Found):
        
        print("\n\n\n\n")
        showTime()
        print("IGLIMIT:%f, ratio:%d"%(IGLIMIT,ratio))
        rootCause = RCA(IGLIMIT,ratio)
        prediction1 = {}
        for item in rootCause[:10]:
            prediction1[item[0]] = item[1]
        for i in range(4):
            rootCause = RCA(IGLIMIT,ratio)
            prediction2 = {}
            for item in rootCause[:]:
                prediction2[item[0]] = item[1]
            prediction1 = intersection(prediction1, prediction2)
            if(len(prediction1)<1): 
                IGLIMIT -= 0.01
                if(IGLIMIT<0.04):
                    print("the model cannot work")
                    return
                break
            if(i==3):
                Found = True
    showTime()
    result = sorted(prediction1.items(), key = lambda x:x[1], reverse=True)
    print("the final result for root cause recommendation")
    for item in result:
        print(item)
    
    return 

if __name__=="__main__":
    
    main()
    
    
    
    


