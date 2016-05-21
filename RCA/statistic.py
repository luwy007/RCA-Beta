# -*- coding=utf-8 -*-
'''
Created on 2016年3月7日

@author: YANG
'''

import os
import xlrd

import matplotlib.pyplot as plt
    
    
    

"145800_ViVo_City"


if __name__=="__main__":
    path = "C:\\Users\YANG\Desktop\Data"    #os.path.dirname("statistic.py")
    book = xlrd.open_workbook("%s\\4.xls"%path)
    sheet = book.sheet_by_index(0)
    dic = {}
    problems = {}
    time = {}
    for rowIndex in range(8,sheet.nrows):
        
        if(sheet.cell_value(rowIndex,2)!="130850_ITS_Centre"):
            continue
            pass
        try:
            dic[sheet.cell_value(rowIndex,2)] += 1
        except:
            dic[sheet.cell_value(rowIndex,2)] = 1
        
        if(not time.__contains__(sheet.cell_value(rowIndex,0))):
            time[sheet.cell_value(rowIndex,0)] = [0,0]
        
        if(not problems.__contains__(sheet.cell_value(rowIndex,2))):
            problems[sheet.cell_value(rowIndex,2)] = 0
        
        if(sheet.cell_value(rowIndex,17)!="NIL" and float(sheet.cell_value(rowIndex,17))<99.5):
            time[sheet.cell_value(rowIndex,0)][0] += 1
            problems[sheet.cell_value(rowIndex,2)] += 1 
        else:
            time[sheet.cell_value(rowIndex,0)][1] += 1
    for item in problems:
        problems[item] /= dic[item]            
    
    sorted_result = sorted(time.items(), key= lambda x:x[0], reverse=True)
    
    
    
    
    for item in sorted_result:
        print(item,time[item[0]][0]/(time[item[0]][1]+1))
        pass
    index = 0
    x = []
    y = []
    for item in sorted_result[-7:]:
        x.append(index)
        index += 1
        y.append(time[item[0]][0]/(time[item[0]][1]+time[item[0]][0]))
        
    for item in sorted_result[:-8]:
        x.append(index)
        index += 1
        y.append(time[item[0]][0]/(time[item[0]][1]+time[item[0]][0]))    
        
    plt.plot(x, y, '-*')
    plt.title('')
    plt.ylabel('ratio')
    plt.xlabel('time')
    plt.show()
     
        
        
        














