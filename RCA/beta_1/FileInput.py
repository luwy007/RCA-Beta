# -*- coding=utf-8 -*-
''' 

@author: YANG
'''

'''
for training, FileInput transform data into two parts, features and labels for train
# this part needs to be considered
for prediction, FileInput transforms data into features


'''

import xlrd
import xlwt
import numpy as np
import os
class FileInput():
    '''
    # 给Train和Predict提供良好的数据读取接口
    # 针对Train，输入是文件路径名，文件名，指定的数据范围，包括哪些列是特征，哪些列是类标
    # 针对Predict，输入是文件路径名，文件名，指定的数据范围，包括哪些列是特征，哪些列是类标？暂不确定
    '''
    def __init__(self):
        pass

    def DataPreprocess(self, fileName = "1"):
        '''
        #对数据进行下述预处理：（去噪，数据转存）
        1）记录包含NIL的列信息，如果NIL个数不超过总样本数的2%，则利用众数代替。否则以全零代替（消除该特征对接下来的模型训练预测产生影响）
        2）将预处理的数据重新写到"tempdata//%s.xls"%fileName文件中
        
        #注意：
        #在dataPreprocess之后，tempdata中的数据应该不存在噪声（即NIL），但是应该存在着众多的“最终无关项”。在beta1版本中需要对此多加注意
        
        '''
        try:
            book = xlrd.open_workbook(fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        
        os.makedirs("tempdata",exist_ok=True)
        book = xlwt.Workbook()
        outputSheet = book.add_sheet("tempSheet")
        
        for colIndex in range(sheet.ncols):
            NILCount = 0
            itemDic = {} #列元素，方便查找众数
            for rowIndex in range(sheet.nrows):
                if(sheet.cell_value(rowIndex,colIndex)=="NIL"):
                    NILCount += 1
                else:
                    try:
                        itemDic[sheet.cell_value(rowIndex,colIndex)] += 1
                    except:
                        itemDic[sheet.cell_value(rowIndex,colIndex)] = 1
            mode = "0\n" # 众数，设置为0，以便于将NIL过多的特征剔除
            

            if(NILCount<sheet.nrows*0.02):
                count = 0
                for item in itemDic:
                    if(itemDic[item]>count):
                        mode = item
                for rowIndex in range(sheet.nrows):
                    if(sheet.cell_value(rowIndex,colIndex)=="NIL"):
                        outputSheet.write(rowIndex,colIndex,mode)
                        continue
                    outputSheet.write(rowIndex,colIndex,sheet.cell_value(rowIndex,colIndex))
            else:
                for rowIndex in range(8):
                    outputSheet.write(rowIndex,colIndex,sheet.cell_value(rowIndex,colIndex))
                for rowIndex in range(8,sheet.nrows):
                    outputSheet.write(rowIndex,colIndex,mode)
                        
        book.save("tempdata//%s.xls"%fileName)
        
    def InputForTrain(self, path="tempdata", fileName="1", rowBegin=8, cols=[-2]*61):
        '''
        #参数说明
        # path is where the training data located
        # fileName is the name of training data
        # rowBegin is the row index from where the data stored
        # cols表示哪些列作为features，哪些作为label，哪些被抛弃，分别用0，1，-1表示。数据总共61列
        
        #注意
        #在区分正负样例时，有诸多问题。譬如划分的标准（严格则会有数据倾斜问题，放宽限制有可能引入噪声）
        '''
        try:
            book = xlrd.open_workbook(path+"//"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        
        features = []
        for colIndex in range(len(cols)):
            if(cols[colIndex]!=0):
                continue
            temp = []
            for rowIndex in range(rowBegin,sheet.nrows):
                try:
                    temp.append(float(sheet.cell_value(rowIndex, colIndex)))
                except:
                    temp.append(0)
                    print("NIL", rowIndex, colIndex) 
            features.append(temp)
        features = np.array(features).transpose()
        
        labelIndex = -1
        for item in cols:
            labelIndex += 1
            if(item==1):
                break
        '''
        #此处对正负类的划分规则，相较于工业上的阈值区分要更加严格。
        #定义当指标为非完美时，便为负类，以此解决数据倾斜问题，并且可以尽可能捕捉到更多的根因识别有效信息
        '''
        labels = []
        if(labelIndex==9):
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item>=2):
                    labels.append(1)
                else:
                    labels.append(-1)
        elif(labelIndex==26):
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item==0):
                    labels.append(1)
                else:
                    labels.append(-1)
        else:
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item==100):
                    labels.append(1)
                else:
                    labels.append(-1)
        
        return features, labels
    
    def InputForPredict(self, path="tempdata", fileName="1", rowBegin=8, cols=[-2]*61):
        '''
        # path is where the training data located
        # fileName is the name of training data
        # rowBegin is the row index from where the real data stored
        # cols表示哪些列作为features，哪些作为label，哪些被抛弃，分别用0，1，-1表示。 数据总共61列
        
        # 此处隐藏的问题和InputForTrain中一致，即究竟如何划分数据的正负类。
        # 划分的松散有可能引入噪声。暂定为按照工业界阈值来进行划分
        '''
        try:
            book = xlrd.open_workbook(path+"//"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        
        labelIndex = -1
        for item in cols:
            labelIndex += 1
            if(item==1):
                break

        labels = []
        if(labelIndex==9):
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item>1):
                    labels.append(1)
                else:
                    labels.append(-1)
        elif(labelIndex==26):
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item<0.5):
                    labels.append(1)
                else:
                    labels.append(-1)
        else:
            for item in sheet.col_values(labelIndex)[rowBegin:]:
                if(item>99.5):
                    labels.append(1)
                else:
                    labels.append(-1)

        features = []
        for rowIndex in range(rowBegin, sheet.nrows):
            if(labels[rowIndex-rowBegin]!=-1):
                continue
            temp = []
            for colIndex in range(len(cols)):
                if(cols[colIndex]!=0):
                    continue
                temp.append(float(sheet.cell_value(rowIndex,colIndex)))
            features.append(temp)
        features = np.array(features)
        return features, labels

    def InputGetDic(self, path="tempdata", fileName="1", nameRow=6, cols=[-2]*61):
        try:
            book = xlrd.open_workbook(path+"//"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        dic = {}
        colIndex = 0
        for i in range(sheet.ncols):
            if(cols[i]!=0):
                continue
            dic[colIndex] = sheet.cell_value(nameRow,i)+"_"+sheet.cell_value(nameRow+1,i)
            colIndex += 1
        return dic


if __name__=="__main__": 
    pass
    
    #obj.DataPreprocess()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




