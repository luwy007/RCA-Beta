# -*- coding=utf-8 -*-
'''
Created on 2016年2月19日

@author: YANG
'''
from Beta_1.FileInput import FileInput 
import os
import pickle
import math
from Beta_1.RCA_Train import RCA_Train 

class RCA_Predict:
    
    def __init__(self, fileName="", label=1):
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
        self.tree = {}
    
    def Predict(self, filteredFea=[]):
        #=======================================================================
        # 为了自适应调整模型得出的结果，添加人为定义的filteredFea，即网络调优工程师分析排除的特征将不出现在下一次模型训练中
        # 增强了模型的适应能力，更符合实际使用时的需求
        #=======================================================================
        cols = RCA_Train().DefineCols(label=self.label, filteredCols=[i for i in range(4)]+filteredFea)
        features, labels = FileInput().InputForPredict(self.fileName,cols)
        reader = open("models\\%s_model"%self.fileName,'rb')
        self.tree = pickle.load(reader)
        reader.close()
        
        IG = {}    # IG = { fea : { pos : count }}
        for feature in features:
            #  path = [[position,node=tree[position]]]
            path = self.FindPath(feature)
            for index in range(len(path)):
                node = path[index][1]
                fea = node[0]
                if(fea==-1):    # fea如果是-1，表示没有选择任何feature作为划分辅助，即该节点为叶节点
                    continue
                
                pos = path[index][0]
                if(not IG.__contains__(fea)):
                    IG[fea] = {pos:0}
                try:
                    IG[fea][pos] += 1
                except:
                    IG[fea][pos] = 1
                
        IGRank = {}
        NEG = 0
        for fea in IG:
            count = 0
            for pos in IG[fea]:
                count += IG[fea][pos]
            tempIG = 0
            for pos in IG[fea]:
                if(self.tree[pos][3]<0):
                    print("IG is negative, %d, %f,"%(pos,self.tree[pos][3]))
                    NEG += 1
                tempIG += self.tree[pos][3]*IG[fea][pos]/count
            IGRank[fea] = tempIG
        
        print("size of the tree is %d"%len(self.tree))
        
        
        result = sorted(IGRank.items(), key = lambda x:x[1], reverse=True)
        
        feaDic = FileInput().InputGetDic(self.fileName, cols)
        rootCause = [] 
        for item in result:
            try:
                rootCause.append([feaDic[item[0]], item[1]]) 
            except:
                print(item[0])
        print("%d skeptical root causes"%len(rootCause))
        
        for item in rootCause[:]:
            print(item)
        
        
        
        IGTotal = 0
        for item in rootCause[:]:
            IGTotal += item[1]
        
        for i in range(1,5):
            temp = 0
            for item in rootCause[:5*i]:
                temp += item[1]
            #print(temp/IGTotal)
        
        return rootCause
        
    def VisualizeTree(self):
        print("nodes of the tree is %i"%len(self.tree))
        height = -1
        for i in self.tree:
            if(i>max):
                height = i 
        print("the height of the tree is %i"%(int(math.log2(height))+1))
        
        i = 1
        height = 0
        while(height<10):
            while(i<2**height):
                if(self.tree.__contains__(i)):
                    print(str(self.tree[i][0])+" ", end="")
                else:
                    print(" ", end="")
                i += 1
            print()
            height += 1    
    
    def FindPath(self, feature):
        '''
        #记录每条item在决策树中游走的过程，记录经过的节点，及各个节点的特性
        '''
        path = []
        position = 1
        node = self.tree[position]
        path.append([position,node])
        while(node[0]!=-1):
            if(feature[node[0]]<node[1]):
                position *= 2
            else:
                position *= 2
                position += 1
            node = self.tree[position]
            path.append([position,node])
        return path
    

    
    
    
    
    
    
    
    
    
    
    