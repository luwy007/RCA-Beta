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
    给Train和Predict提供良好的数据读取接口
    针对Train，输入是文件路径名，文件名，指定的数据范围，包括哪些列是特征，哪些列是类标
    针对Predict，输入是文件路径名，文件名，指定的数据范围，包括哪些列是特征，哪些列是类标？暂不确定
    '''
    def __init__(self):
        pass

    def DataPreprocess(self, fileName = "sample data"):
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
            mode = "" #众数
            if(NILCount>sheet.nrows*0.02):
                count = 0
                for item in itemDic:
                    if(itemDic[item]>count):
                        mode = item
            for rowIndex in range(sheet.nrows):
                if(sheet.cell_value(rowIndex,colIndex)=="NIL"):
                    outputSheet.write(rowIndex,colIndex,mode)
                    continue
                outputSheet.write(rowIndex,colIndex,sheet.cell_value(rowIndex,colIndex))
        
        book.save("tempdata//%s.xls"%fileName)

    def InputForTrain(self, path="tempdata", fileName="1", rowBegin=8, cols=[-2]*61):
        '''
        参数说明
        # path is where the training data located
        # fileName is the name of training data
        # rowBegin is the row index from where the real data stored
        # cols表示哪些列作为features，哪些作为label，哪些被抛弃，分别用0，1，-1表示。2.0版本数据总共61列
        '''
        try:
            book = xlrd.open_workbook(path+"\\"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        
        featuresDic = {}
        for rowIndex in range(rowBegin,sheet.nrows):
            temp = []
            for colIndex in range(len(cols)):
                if(cols[colIndex]!=0):
                    continue
                try:
                    temp.append(float(sheet.cell_value(rowIndex,colIndex)))
                except:
                    temp.append(0)
                    print("NIL ERROR %i %i"%(rowIndex,colIndex))
            try:
                featuresDic[sheet.cell_value(rowIndex,2)].append(temp)
            except:
                featuresDic[sheet.cell_value(rowIndex,2)] = [temp]

        for item in featuresDic:
            featuresDic[item] = np.array(featuresDic[item])



        
        labelIndex = -1
        for item in cols:
            labelIndex += 1
            if(item==1):
                break
        '''
        此处对正负类的划分规则，相较于工业上的阈值区分要更加严格。
        定义当指标为非完美时，便为负类，以此解决数据倾斜问题，并且可以尽可能捕捉到更多的根因识别有效信息
        '''
        labelsDic = {}
        for rowIndex in range(rowBegin,sheet.nrows):
            label = -1
            if(sheet.cell_value(rowIndex,labelIndex)==100):
                label = 1
            try:
                labelsDic[sheet.cell_value(rowIndex,2)].append(label)
            except:
                labelsDic[sheet.cell_value(rowIndex,2)] = [label]

        return featuresDic, labelsDic

    def InputForPredict(self, path="tempdata", fileName="1", rowBegin=8, cols=[-2]*61):
        '''
        参数说明
        # path is where the training data located
        # fileName is the name of training data
        # rowBegin is the row index from where the real data stored
        # cols表示哪些列作为features，哪些作为label，哪些被抛弃，分别用0，1，-1表示。2.0版本数据总共61列
        '''
        try:
            book = xlrd.open_workbook(path+"\\"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        
        featuresDic = {}
        for rowIndex in range(rowBegin,sheet.nrows):
            temp = []
            for colIndex in range(len(cols)):
                if(cols[colIndex]!=0):
                    continue
                try:
                    temp.append(float(sheet.cell_value(rowIndex,colIndex)))
                except:
                    temp.append(0)
                    print("NIL ERROR %i %i"%(rowIndex,colIndex))
            try:
                featuresDic[sheet.cell_value(rowIndex,2)].append(temp)
            except:
                featuresDic[sheet.cell_value(rowIndex,2)] = [temp]

        for item in featuresDic:
            featuresDic[item] = np.array(featuresDic[item])



        
        labelIndex = -1
        for item in cols:
            labelIndex += 1
            if(item==1):
                break
        '''
        此处对正负类的划分规则，相较于工业上的阈值区分要更加严格。
        定义当指标为非完美时，便为负类，以此解决数据倾斜问题，并且可以尽可能捕捉到更多的根因识别有效信息
        '''
        labelsDic = {}
        for rowIndex in range(rowBegin,sheet.nrows):
            label = -1
            if(sheet.cell_value(rowIndex,labelIndex)>99.5):
                label = 1
            try:
                labelsDic[sheet.cell_value(rowIndex,2)].append(label)
            except:
                labelsDic[sheet.cell_value(rowIndex,2)] = [label]

        return featuresDic, labelsDic

    def InputGetDic(self, path="tempdata", fileName="1", nameRow=6, cols=[-2]*61):
        try:
            book = xlrd.open_workbook(path+"\\"+fileName+".xls")
        except Exception as e:
            print(e)
            return 
        sheet = book.sheet_by_index(0)
        dic = {}
        colIndex = 0
        for i in range(sheet.ncols):
            if(cols[i]!=0):
                continue
            dic[colIndex] = sheet.cell_value(nameRow,i)
            colIndex += 1
        return dic


if __name__=="__main__": 
    obj = FileInput()
    obj.DataPreprocess()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




