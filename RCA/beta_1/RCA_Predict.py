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
    
    def __init__(self):
        '''
        tree = { Position : [selectedAttr, splitPoint, entropy (itself, leftSon, rightSon),IG] }
        # Position 是各个节点在决策树中所处的位置， 根节点处于位置 “1”
        # selectedAttr 表示该节点选取的属性，以特征下标表示，而非名称
        # splitPoint 表示以selectedAttr为判别属性时，分界属性值是什么
        # entropy表明在该节点处所有统计数据的熵情况
        # IG information gain of the node
        '''
        self.tree = {}
    
    def Predict(self, path="tempdata", fileName="1"):
    
        cols = RCA_Train().DefineCols(label=17, filteredCols=[i for i in range(4)])
        features, labels = FileInput().InputForPredict(cols=cols)
        reader = open(path+"\\"+fileName,'rb')
        self.tree = pickle.load(reader)
        reader.close()
        
        
        IG = {}    # IG = { fea : { pos : count }}
        for feature in features:
            #  path = [[position,node=tree[position]]]
            path = self.FindPath(feature)
            for index in range(len(path)):
                node = path[index][1]
                fea = node[0]
                pos = path[index][0]
                if(not IG.__contains__(fea)):
                    IG[fea] = {pos:0}
                try:
                    IG[fea][pos] += 1
                except:
                    IG[fea][pos] = 1
                
        IGRank = {}
        for fea in IG:
            count = 0
            for pos in IG[fea]:
                count += IG[fea][pos]
            tempIG = 0
            for pos in IG[fea]:
                tempIG += self.tree[pos][3]*IG[fea][pos]/count
            IGRank[fea] = tempIG
        
        result = sorted(IGRank.items(), key = lambda x:x[1], reverse=True)
        
        feaDic = FileInput().InputGetDic()
        rootCause = []
        for item in result:
            rootCause.append([feaDic[item[0]],item[1]])
        print(rootCause)
        
        return 
        '''
        # 接下来的部分，主要用于展示已经生成的决策树模型。 在beta版本中并不需要
        '''
        
        leafNodeDic = {} 
        pathes = []
        for index in range(len(labels)):
            if(labels[index]==-1):
                path = self.FindPath(features[index])
                pathes.append(path)
                continue
                for item in path:
                    print(item[0], end=" ")
                print()
                try:
                    leafNodeDic[path[-1][0]] += 1
                except:
                    leafNodeDic[path[-1][0]] = 1
        treeStrut = []
        for path in pathes:
            for item in path:
                try:
                    treeStrut[int(math.log2(item[0]))][item[0]] += 1
                except:
                    try:
                        treeStrut[int(math.log2(item[0]))][item[0]] = 1
                    except:
                        treeStrut.append({})
                        treeStrut[int(math.log2(item[0]))][item[0]] = 1
                    
        for index in range(len(treeStrut)):
            dic = treeStrut[index]
            l = sorted(dic.items(), key=lambda x:x[0])
            print("                           ",end="")
            for item in l:
                print(item,end = "")
            print()
    
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
    
if __name__=="__main__":
    ig = {1:2,2:3}
    result = sorted(ig.items(), key = lambda x:x[1])
    RCA_Predict().Predict()  
    
    
    
    
    
    
    
    
    
    
    
    