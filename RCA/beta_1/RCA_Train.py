# -*- coding=utf-8 -*-
'''
Created on 2016/2/18

@author: YANG
'''
from Beta_1.FileInput import FileInput
import math
import pickle
import os
import random
import numpy as np
import time

ENTROPYLIMIT = 0.1

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

'''
    cellId in 1.xls
    3 206 0.01456 132180_Ang_Mo_Kio_St_31_Blk_319
    8 412 0.01942 142560_Ang_Mo_Kio_St_22_Blk_207
    5 412 0.01214 146150_Bishan_St_23_Blk_288
    2 206 0.00971 144760_Sin_Ming_Autocare
    5 206 0.02427 130760_Ang_Mo_Kio_Ave_10_Blk_405
    45 412 0.10922 146680_Ang_Mo_Kio_Ave_1_Blk_331 

'''



class RCA_Train():
     
    
    def __init__(self, fileName="", label=1, IGLIMIT=0, ratio=1):
        '''
        tree = { Position : [selectedAttr, splitPoint, entropy (itself, leftSon, rightSon),IG] }
        # Position 是各个节点在决策树中所处的位置， 根节点处于位置 “1”
        # selectedAttr 表示该节点选取的属性，以特征下标表示，而非名称
        # splitPoint 表示以selectedAttr为判别属性时，分界属性值是什么
        # entropy表明在该节点处所有统计数据的熵情况
        # IG information gain of the node
        '''
        self.label = label
        self.fileName = fileName
        self.IGLIMIT = IGLIMIT
        self.ratio = ratio
        self.tree = {}
    
    def AdjustPosNeg(self, features, labels, ratio):
        '''
        ratio = Pos:Neg
        '''
        tempF = np.array([0]*features.shape[1])
        tempL = []
        negCount = 0
        for item in labels:
            if(item==-1):
                negCount += 1
        
        Limit = ratio*negCount/(len(labels)-negCount)
        
        for index, item in enumerate(labels):
            if(item==1 and random.random()>Limit):
                continue
            tempL.append(item)
            tempF = np.vstack([tempF,features[index,:]])
        
        tempF = np.array(tempF[1:,:])
        
        return tempF, tempL
    
    def Train(self, filteredFea=[]):
        '''
        # 训练模型&保存
        # 虽然predict做的工作较少，只是读取训练的模型，结合负样本特征，给出最终预测。
        # 但是为了逻辑上的清晰，还是将predict功能单独辟出来
        # 为了自适应调整模型得出的结果，添加人为定义的filteredFea，即网络调优工程师分析排除的特征将不出现在下一次模型训练中
        # 增强了模型的适应能力，更符合实际使用时的需求
        '''
        cols = self.DefineCols(label=self.label, filteredCols=[i for i in range(4)]+filteredFea)
        features, labels = FileInput().InputForTrain(self.fileName, cols)
        features, labels = self.AdjustPosNeg(features,labels,self.ratio)
        self.tree = {}
        self.BuildTree(features, labels)
        os.makedirs("models", exist_ok=True)
        writer = open("models\\%s_model"%self.fileName,'wb')
        pickle.dump(self.tree, writer)
        writer.close()
        
    def DefineCols(self, label=0, filteredCols = []):
        #将filteredCols中提及的列设置为-1，label代表的列设置为1，其余为0
        
        KPIs = [9,17,25,26,27]   # 过滤掉所有重点监测的其他KPI，这些是可以事先排除的因素
        cols = [0]*61
        for index in filteredCols+KPIs:
            cols[index] = -1
        cols[label] = 1
        return cols

    def BuildTree(self, features, labels):
        filteredFea = [0]*len(features[0])
        filteredSam = [0]*len(labels)
        # ChooseBestAttr返回的第一个参数是三元tuple，注意！     
        rootEntropy, rootAttr, rootSplitPoint = self.ChooseBestAttr(features,labels,filteredFea,filteredSam)
        
      
        NEGCount = 0
        for item in labels:
            if(item==-1):
                NEGCount += 1
        p = NEGCount/len(labels)
        initEntropy = -(p*math.log2(p)+(1-p)*math.log2(1-p)) #计算原始样本中的熵，用于计算之后的熵增益
        IG = initEntropy-rootEntropy[0]  #information gain of the root
        self.tree[1] = [rootAttr, rootSplitPoint, rootEntropy, IG]
        filteredFea[rootAttr] = 1
        isLeaf = False
        if(IG<self.IGLIMIT):
            isLeaf = True
        self.TreeGrowth(features, labels, filteredFea, filteredSam, 1, True, isLeaf)
        
        if(IG<self.IGLIMIT):
            isLeaf = True
        self.TreeGrowth(features, labels, filteredFea, filteredSam, 1, False, isLeaf)
 
    def TreeGrowth(self, features, labels, filteredFea, filteredSam, parentPosition, isLeft, isLeaf):
        '''
        #如果是计算父节点时发现此节点不添加任何判断属性，熵就已经达标的话，便将此节点定义为叶节点
        '''
        if(isLeaf): 
            if(isLeft):
                entropy = [self.tree[parentPosition][2][1],-1,-1]
                self.tree[2*parentPosition] = [-1, -1, entropy, 0]    # 叶节点未能给分类带来帮助， IG = 0
            else:
                entropy = [self.tree[parentPosition][2][2],-1,-1]
                self.tree[2*parentPosition+1] = [-1, -1, entropy, 0]
            return


        parentSelectedAttr = self.tree[parentPosition][0]
        parentSplitPoint = self.tree[parentPosition][1]
        
        if(isLeft):
            leftFilteredSam = filteredSam[:]
            leftFilteredFea = filteredFea[:]
            for i in range(len(labels)):
                if(features[i][parentSelectedAttr]>=parentSplitPoint):
                    leftFilteredSam[i] = 1

            entropy, attr, splitPoint = self.ChooseBestAttr(features,labels,filteredFea,leftFilteredSam)  
            IG = self.tree[parentPosition][2][1] - entropy[0]
            self.tree[2*parentPosition] = [attr, splitPoint, entropy, IG]
            leftFilteredFea[attr] = 1
            self.TreeGrowth(features, labels, leftFilteredFea, leftFilteredSam, 2*parentPosition, True, IG<self.IGLIMIT)
            self.TreeGrowth(features, labels, leftFilteredFea, leftFilteredSam, 2*parentPosition, False, IG<self.IGLIMIT)
        else:
            rightFilteredSam = filteredSam[:]
            rightFilteredFea = filteredFea[:]
            for i in range(len(labels)):
                if(features[i][parentSelectedAttr]<parentSplitPoint):
                    rightFilteredSam[i] = 1

            entropy, attr, splitPoint = self.ChooseBestAttr(features,labels,filteredFea,rightFilteredSam)
            IG = self.tree[parentPosition][2][2] - entropy[0]   
            self.tree[2*parentPosition+1] = [attr, splitPoint, entropy, IG]
            rightFilteredFea[attr] = 1
            self.TreeGrowth(features, labels, rightFilteredFea, rightFilteredSam, 2*parentPosition+1, True, IG<self.IGLIMIT)
            self.TreeGrowth(features, labels, rightFilteredFea, rightFilteredSam, 2*parentPosition+1, False, IG<self.IGLIMIT)

        return
 
    def ChooseBestAttr(self, features, labels, filteredFea, filteredSam):
        '''
        features contains several features, labels contains +1 or -1, 
        selected represents whether the feature selected, 0 not selected, 1 selected
        filteredFea 1表示已经在之前的节点中选过了
        filteredSam 1表示已经在之前的节点中被过滤掉了
        '''
        MinEntropy = [1,0,0]
        MinEntropyIndex = -1
        MinEntropySplitPoint = -1
        
        for i in range(len(filteredFea)):
            if(filteredFea[i]==1):
                continue
            tempData = []
            tempLabels = []
            for index in range(len(filteredSam)):
                if(filteredSam[index]==1):
                    continue
                tempData.append(features[index][i])
                tempLabels.append(labels[index])
            Entropy,SplitPoint = self.GetEntropyAndSplitPoint(tempData, tempLabels)
            if(Entropy[0]<=MinEntropy[0]):
                MinEntropy = Entropy
                MinEntropyIndex = i
                MinEntropySplitPoint = SplitPoint
                
        return MinEntropy, MinEntropyIndex, MinEntropySplitPoint

    def GetEntropyAndSplitPoint(self, feature, labels):
        #feature is a 1-dimension vector, |feature|=|labels|,labels contains +1 or -1
        entropyDic = {}
        minEntropy = [1,0,0]
        minEntropySplit = 0
        
        for splitPoint in feature:
            if(splitPoint in entropyDic.keys()):
                continue
            else:
                entropy = self.EntropyCalculation(feature, labels, splitPoint)
                if(entropy[0]<minEntropy[0]):
                    minEntropy = entropy
                    minEntropySplit = splitPoint
                entropyDic[splitPoint] = entropy
        
        return minEntropy,minEntropySplit
        
    def EntropyCalculation(self,feature,labels,splitPoint):
        #feature is a 1-dimension vector, |feature|=|labels|, splitPoint is the boundary in feature
        if(len(labels)==0):
            print("entropy calculation error! feature has no elements")
            return -1
        
        YPosXLarger = 0
        YPosXSmaller = 0
        YNegXLarger = 0
        YNegXSmaller = 0 
        
        for index,item in enumerate(feature):
            if(labels[index]==1):
                if(item>=splitPoint):
                    YPosXLarger += 1
                else:
                    YPosXSmaller += 1
            else:
                if(item>=splitPoint):
                    YNegXLarger += 1
                else:
                    YNegXSmaller += 1
        
        LargerSum = YPosXLarger+YNegXLarger
        SmallerSum = YPosXSmaller+YNegXSmaller
        LargerEntropy = 0
        SmallerEntropy = 0
        
        if(LargerSum==0):
            pass
        else:
            if(YPosXLarger==0):
                pass
            else:
                p = YPosXLarger/LargerSum
                LargerEntropy -= p*math.log2(p)
            
            if(YNegXLarger==0):
                pass
            else:
                p = YNegXLarger/LargerSum
                LargerEntropy -= p*math.log2(p)
        
        if(SmallerSum==0):
            pass
        else:
            if(YPosXSmaller==0):
                pass
            else:
                p = YPosXSmaller/SmallerSum
                SmallerEntropy -= p*math.log2(p)
            
            if(YNegXSmaller==0):
                pass
            else:
                p = YNegXSmaller/SmallerSum
                SmallerEntropy -= p*math.log2(p)

        entropy = (LargerSum/(LargerSum+SmallerSum))*LargerEntropy+\
                  (SmallerSum/(LargerSum+SmallerSum))*SmallerEntropy
        
        return entropy, SmallerEntropy, LargerEntropy
 
    def Pruning(self):
        
        pass
    
     
     




















