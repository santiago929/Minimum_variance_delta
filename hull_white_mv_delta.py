# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 09:09:35 2023

@author: Santiago

"""
import pandas as pd
import numpy as np
import statsmodels.api as sm 
import datetime
from datetime import timedelta
from tabulate import tabulate

def train_data(df, startIdx, windowLength, isNormalized):
    tmp = df.copy()
    tmp.reset_index(inplace = True)
    train_start_index = startIdx
    train_window_length = windowLength
    train_end_index = train_start_index+train_window_length
    tmp = tmp.iloc[train_start_index:train_end_index]

    if(isNormalized==False):
        tmp['opt_price_chg'] = tmp['c'].diff()
        tmp['umid_chg'] = tmp['umid'].diff()
        tmp = tmp[tmp['umid_chg']!=0] 
    else:
        tmp['opt_price_chg'] = tmp['c'] / tmp['c'].shift()
        tmp['umid_chg'] = tmp['umid'] / tmp['umid'].shift()
        tmp = tmp[tmp['umid_chg']!=1]
    
    tmp = tmp[['date','opt_price_chg','umid_chg','delta','vega','dte','umid']]
    tmp.reset_index(drop=True, inplace=True)
    train_set = pd.DataFrame()
    coeff = tmp['vega']*tmp['umid_chg']/tmp['umid']/np.sqrt(tmp['dte']/252)
    train_set['y'] = (tmp['opt_price_chg'] - tmp['delta']*tmp['umid_chg'])
    train_set['x1'] = coeff*tmp['delta']
    train_set['x2'] = coeff*tmp['delta']**2
    train_set = train_set.join(tmp)
    train_set.drop(index=0, inplace=True)
    
    return train_set

def train_model(train_set):
    x = sm.add_constant(train_set.loc[:,['x1','x2']])
    y = train_set['y']
    model = sm.OLS(y,x)
    result = model.fit()
    #se = (result.resid)**2
    #sse = np.sum(se)
    return result

def quadratic_fit(df, startIdx, windowLength, result):
    tmp = df.copy()
    test_start_index = startIdx
    test_window_length = windowLength
    test_end_index = test_start_index + test_window_length
    tmp = tmp.iloc[test_start_index:test_end_index]
    tmp = tmp[['date','delta','vega','dte','umid', 'c']]
    a = result.params[0]
    b = result.params[1]
    c = result.params[2]
    #Quadratic fit
    tmp['y_hat'] = a + b*tmp['delta'] + c*tmp['delta']**2
    tmp['MV_delta'] = tmp['delta'] +tmp['vega']*tmp['y_hat']/tmp['umid']/np.sqrt(tmp['dte']/252) #Hull&White Eq(2.a)
    tmp['R^2'] = result.rsquared
    tmp['a_hat'] = a
    tmp['b_hat'] = b
    tmp['c_hat'] = c
    tmp.reset_index(inplace=True)
    
    return tmp

def generate_MV_delta(df, trainStartIdx, trainLength, isNormalized, quadratic_fit_length):
    train_set = train_data(df, trainStartIdx, trainLength, isNormalized)
    result = train_model(train_set)
    #print(result.summary())
    predict_set = quadratic_fit(df, trainStartIdx + trainLength, quadratic_fit_length,result)  
    
    return predict_set

def gain_test(df, isNormalized):
    if(isNormalized==False):
        df['opt_price_chg'] = df['c'].diff()
        df['umid_chg'] = df['umid'].diff()
        df = df[df['umid_chg']!=0] 
    else:
        df['opt_price_chg'] = df['c'] / df['c'].shift()
        df['umid_chg'] = df['umid'] / df['umid'].shift()
        df = df[df['umid_chg']!=1]
        
    gain_df = df[['delta', 'MV_delta', 'opt_price_chg', 'umid_chg']]  
    gain_df['E_mv'] = 0
    gain_df['E_bs'] = 0
    
    for i in range(len(gain_df.index)):
        gain_df['E_mv'].iloc[i] = gain_df['opt_price_chg'].iloc[i] - \
            gain_df['MV_delta'].iloc[i-1]*gain_df['umid_chg'].iloc[i]
        gain_df['E_bs'].iloc[i] = gain_df['opt_price_chg'].iloc[i] - \
            gain_df['delta'].iloc[i-1]*gain_df['umid_chg'].iloc[i]    
    
    gain_ratio = 1-(np.sum((gain_df.E_mv)**2)/np.sum((gain_df.E_bs)**2))

    return gain_ratio

def get_expiries(prices: pd.DataFrame) -> dict[np.datetime64, np.ndarray]:
    _expiries = prices[['date', 'expiry']].sort_values(by=['date', 'expiry']).drop_duplicates()
    dates = np.unique(_expiries.date.values.astype('M8[D]'))
    expiry = _expiries.expiry.values.astype('M8[D]')  
    expiries: dict[np.datetime64, np.ndarray] = {}
    for date in dates:
        expiries[date] = expiry[_expiries.date == date]
        
    return expiries

path = ''
spy_df = pd.read_csv(path+'spy_new.csv')
spy_df['timestamp'] = [ datetime.datetime.strptime(spy_df['date'][x], '%d/%m/%Y %H:%M') for x in range(0, len(spy_df.date))]
spy_df['date'] = spy_df.timestamp.values.astype('M8[D]')
spy_df['vega'] = spy_df['vega']*100 
splits = spy_df.symbol.str.split('-', n=2, expand=True)
spy_df['put_call'] = splits[0]
spy_df['strike'] = splits[1].astype(int)
spy_df['expiry'] = pd.to_datetime(splits[2]).values.astype('M8[D]')
spy_df['dte'] = spy_df.expiry - spy_df.date
spy_df['dte'] = spy_df['dte'].dt.days
spy_df['mv_delta'] = 0

startdate = np.datetime64('2024-02-20')
finaldate = np.datetime64('2024-03-15')
expiries = get_expiries(spy_df)
expiry_filter = [pd.Timestamp(x) for x in expiries[startdate]]
contract_filter = ['C', 'P']
trainlength = 4
#strikes_filter: Avoids calculating the mv delta for otm strikes to shorten the execution time.
while startdate < finaldate:
    strikes_filter = np.unique(round(spy_df.loc[spy_df.date == pd.Timestamp(startdate),'umid'])) 
    for c_p in contract_filter:
        for K in strikes_filter:
            for exp in expiry_filter:  
                copy_df = spy_df[(spy_df.date == pd.Timestamp(startdate)) & (spy_df.strike == K) \
                             & (spy_df.expiry == exp) & (spy_df.put_call == c_p)]
                for i in range(trainlength, len(copy_df.index)):
                    mv_delta = generate_MV_delta(copy_df, 0, i, False, 1)
                    mv_delta.set_index('index', inplace=True)
                    spy_df['mv_delta'][mv_delta.index] = mv_delta.MV_delta.values[0]      
    startdate = startdate + np.timedelta64(1, 'D')