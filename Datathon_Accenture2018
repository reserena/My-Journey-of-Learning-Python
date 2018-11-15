# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 23:14:54 2018

@author: serena peng
"""
'''Project Overview:
Predicting 2018 Q1 and Q2 employee salary
Database: LA City Empoyee Payroll recoard 2013-2017
'''

import pandas as pd
import numpy as np

data = pd.read_csv("City_Employee_Payroll.csv")

#fixing the issues in MOU column in original data
data_fixed = data.copy()
data_fixed['MOU'] = data_fixed['MOU'].fillna('7777m')
new_mou = [str(round(m)).zfill(2) if type(m) == float else m for m in data_fixed['MOU']]
new_mou = []
for m in data_fixed['MOU']:
    if (type(m) == float) or (type(m) == int):
        new_mou.append(str(round(m)).zfill(2))
    else:
        new_mou.append(m.zfill(2))

data_fixed['MOU'] = new_mou

#deal with missing values
data_fixed['Benefits Plan'] = data_fixed['Benefits Plan'].fillna('7777b')

data_ml = data_fixed.copy()

#reshape all quarterly payments data
y_col = ['Q1 Payments','Q2 Payments', 'Q3 Payments', 'Q4 Payments']
data_learn = data_fixed[data_fixed['Year']!=2018]
data_pt = pd.pivot_table(data_learn, values = y_col, 
                         index = 'Record Number',columns = 'Year',fill_value = 0)


#remove all duplicate records in the same year
data_ml_uni = data_ml.drop_duplicates(subset=['Record Number', 'Year'])

##Building the training data
##add the time-lag payments into the current dataset
idx = pd.IndexSlice
pmt_df = pd.DataFrame()
x_selected = ['Record Number','Year','Employment Type','Benefits Plan','MOU']
newcol = ['LastYear','y']
for i, j in data_pt.columns: #i = [q1,q2..] and j = [2013,2014]
    if (i == 'Q3 Payments' or i == 'Q4 Payments') and (j == 2017):    
        continue
    if j+1 != 2018:
        test = data_pt.loc[:,idx[i,[j,j+1]]]
    else:
        test = pd.DataFrame(data_pt.loc[:,idx[i,j]])
        test[(i,j+1)]=np.nan
    test.columns = newcol
    test = test.reset_index()
    xdf=data_ml_uni[data_ml_uni['Year']==j+1].filter(x_selected)
    test_x = pd.merge(xdf, test, how = 'left', on = 'Record Number')
    pmt_df = pmt_df.append(test_x)

pmt_df = pmt_df.set_index('Record Number')

##starting to build x and y training data
dumcol = ['Employment Type','Benefits Plan','MOU']
pdf_dum = pmt_df.copy()
pdf_dum.loc[:,newcol]=pdf_dum.loc[:,newcol].fillna(0)
pdf_dum = pd.get_dummies(pdf_dum, columns = dumcol)
x = pdf_dum[pdf_dum['Year']!=2018].drop(['Year','y'],1)
y = pdf_dum[pdf_dum['Year']!=2018].filter('y')
y = pd.Series(y['y'],index = y.index)

pred_x = pdf_dum[pdf_dum['Year']==2018].drop(['Year','y'],1)

####------------------------machine learning-------------------------####
###------------------Step 1: Split data------------------------------###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

###-------------Step 2: starting training-----------------------------###
##------------------Gradient Boost----------------------------------##
from sklearn.ensemble import GradientBoostingRegressor

clf2 = GradientBoostingRegressor().fit(x_train, y_train)
score_gb = clf2.score(x_test, y_test)

'''I tested three different machine algorithms and evaluated their results.
I chose Gradient Boosted Decision Tree because it yielded the best accuracy
based on various measures'''

#predict 2018 payments data
pred_y = clf2.predict(pred_x)

#combine the 2018 q1 and q2 prediction data
pred_y = pd.DataFrame(pred_y,index = pred_x.index).reset_index()\
                     .groupby('Record Number')[0].apply(sum).reset_index()

#export the prediction data into the file
finalfile = pd.read_csv("payroll_2018.csv")
finalfile.columns = ['Record Number',0]
finalfile = finalfile.drop_duplicates(subset = 'Record Number')
finalfile['Record Number'] = finalfile['Record Number'].apply(round).astype(str)
pred_y['Record Number'] = pred_y['Record Number'].astype(str)
final = pd.merge(finalfile.drop(0,1), pred_y, on = 'Record Number',how = 'left')
records = pd.to_numeric(final['Record Number'])
final['Record Number'] = records
final.columns = ['Record.Number', 'Q1.Q2Payments.2018']

'''While I was trying to export data into the csv file provided, I discovered
the file has many duplicate record numbers and also records that does not
exist in the 2018 employee file. Therefore, I was not able to make predictions
for the record numbers that I do not have information for'''
