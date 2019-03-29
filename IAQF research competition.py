# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:54:04 2019

@author: Serena
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import origin
import itertools

class tradingdates:
    
    def __init__(self, df, date_col = None):
        try:
            self.df = df.sort_values(by = df.columns[0])
        except:
            print("Input df is not dataframe")
            return None
        
        self.dcol = date_col
    
    def holidays(self):
        '''a is the start date; b is the end date
        this function return a dataset with all holiday dates'''
        from dateutil import rrule
        
        if self.dcol == None:
            a = self.df.iloc[0,0]
            b = self.df.iloc[-1,0]
            try:
                a.week
            except:
                print("Default DATE column must be the first column")
                return None
        elif type(self.dcol) == str:
            try:
                a = min(self.df[self.dcol])
                b = max(self.df[self.dcol])
            except:
                print("date_col does not exist in df")
                return None
        else:
            print("Invalid date_col argument. Must be str")
            return None
            
        rs = rrule.rruleset()
    
        # Include all potential holiday observances
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 1, bymonthday= 1))                     # New Years Day  
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 1, byweekday= rrule.MO(3)))            # Martin Luther King Day   
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 2, byweekday= rrule.MO(3)))            # Washington's Birthday
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, byeaster= -2))                                  # Good Friday
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 5, byweekday= rrule.MO(-1)))           # Memorial Day
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 7, bymonthday= 4))                     # Independence Day
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth= 9, byweekday= rrule.MO(1)))            # Labor Day
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=11, byweekday= rrule.TH(4)))            # Thanksgiving Day
        rs.rrule(rrule.rrule(rrule.YEARLY, dtstart=a, until=b, bymonth=12, bymonthday=25))                     # Christmas  
        
        # Exclude potential holidays that fall on weekends
        rs.exrule(rrule.rrule(rrule.WEEKLY, dtstart=a, until=b, byweekday=(rrule.SA,rrule.SU)))
    
        return rs

#    def filled(self, missing = None, isna = False):
#        rs = list(self.nontradingdays(self))
#        
#        if self.dcol == None:
#            col = self.df.columns[0]
#        else:
#            col = self.dcol
#        
#        if missing != None:
#            m = missing
#        
#        holidays = []
#        for i in range(0, len(self.df)):
#            if self.df.loc[i, col] in rs:
#                holidays.append(i)
#            elif 
                
def multimerge(dfs, on, how):
    
    if type(dfs) != list:
        print("Invalid dfs argument. Must be a list of dfs")
        return None
    
    if len(dfs) < 3:
        print("Less than 2 df to merge")
        return None
    
    try:
        merged = pd.merge(dfs[0], dfs[1], on = on, how = how)
        for d in dfs[2:]:
            merged = pd.merge(merged, d, on = on, how = how)
        
        return merged
    except:
        print("Invalid on argument")
        return None
    
if __name__ == "__main__":
    path = "S://UCLA//IAQF//data - replicating paper//"
    bindex = pd.read_csv(path + "BAMLC0A1CAAAEY.csv") #AAA bond index
    
    t10 = pd.read_csv(path +"DGS10.csv") #10-year treasury
    t5 = pd.read_csv(path+"DGS5.csv")
    ycurve = pd.read_csv(path +"T10Y2Y.csv")
    lever1 = pd.read_csv(path +"NFCILEVERAGE.csv")
    lever1= pd.DataFrame([['1996-12-27',-0.63]], columns = lever1.columns
                         ).append(lever1) #mannually fill out missing data
    lever2 = pd.read_csv(path +"NFCINONFINLEVERAGE.csv")
    lever2= pd.DataFrame([['1996-12-27',-0.19]], columns = lever2.columns
                         ).append(lever2) #mannually fill out missing data
    vix = pd.read_csv(path +"VIXCLS.csv")
#    sp500 = pd.read_csv(path + "s&p500.csv")
    sp500 = pd.read_csv(path +'^GSPC.csv').iloc[:,[0,5]].rename(columns = {'Date': 'DATE'})
    sp500['sp500'] = sp500['Adj Close']/sp500['Adj Close'].shift()-1
    #sp500['DATE'] = pd.to_datetime(sp500['DATE'])
    #spvix = pd.merge(sp500, factors[['DATE', 'VIXCLS']], on = 'DATE', how = 'left', sort = True)
    
    #spvix = spvix.dropna()
    stressind = pd.read_csv(path +"STLFSI.csv")
    stressind = pd.DataFrame([['1996-12-27',-0.651]], columns = stressind.columns).append(stressind)        
    sentiment = pd.read_csv(path +"UMCSENT.csv")
    leading = pd.read_csv(path +"USSLIND.csv")
    fama = pd.read_csv(path +'fama.csv').rename(columns = {'date': 'DATE'})
    fama['DATE'] = pd.to_datetime(fama['DATE'].astype('str')) #fama french model
    mutual = pd.read_excel(path + 'mutual fund.xlsx').rename(columns = {'date': 'DATE'})
    vwretd = pd.read_csv(path+'vwretd.csv').iloc[:,:2]
    vwretd['DATE'] = pd.to_datetime(vwretd['DATE'].astype(str))
    ted = pd.read_csv(path + 'TEDRATE.csv')
    libor1y = pd.read_csv(path + 'USD12MD156N.csv')    
    t1y = pd.read_csv(path+'DTB1YR.csv')
    swap = pd.read_excel(path + '2yminus5yswap.xlsx').sort_values('Date').rename(columns = {'Date': 'DATE'})
    t3m = pd.read_csv(path + 'DGS3MO.csv')
    
    
    #trading date data
    trading = pd.to_datetime(pd.read_csv(path + "^GSPC.csv")['Date'])
    
    
    '''Construct spread data'''
    spread = pd.merge(bindex, t10, on = 'DATE', how = 'left').dropna().reset_index(drop = True)
    spread['DATE'] = pd.to_datetime(spread['DATE'])
    spread = spread.sort_values(by = 'DATE')
    for c in range(1, len(spread.columns)):
        spread.iloc[:,c] = spread.iloc[:,c].replace('.', value = None, method = 'pad')
    spread.iloc[:,1:] = spread.iloc[:,1:].apply(pd.to_numeric)
    
    holidays = list(tradingdates(spread).holidays())
    h = []
    for i in range(0, len(spread)):
        if spread.iloc[i,0] in holidays:
            h.append(i)
    
    spread_f = spread.drop(h).reset_index(drop = True)
    spread_f['spread'] = spread_f.iloc[:,1] - spread_f.iloc[:,2]
 
    
    '''Merge the factor data'''
    '''1. Daily frequency'''
    factor_daily = multimerge([t10, t5, t3m, ycurve, vix, ted, libor1y, t1y], 
                              on = 'DATE', how = 'outer')
    factor_daily['DATE'] = pd.to_datetime(factor_daily['DATE'])
        
    for c in range(1, len(factor_daily.columns)):
        factor_daily.iloc[:,c] = factor_daily.iloc[:,c].replace('.', value = None, method = 'pad')  
        
    factor_daily.iloc[:,1:] = factor_daily.iloc[:,1:].apply(pd.to_numeric)
    factor_daily['ted1y'] = factor_daily['USD12MD156N'] - factor_daily['DTB1YR']
    factor_daily['T10-T5'] = factor_daily['DGS10']-factor_daily['DGS5'] 
    factor_daily['T10-3'] = factor_daily['DGS10']-factor_daily['DGS3MO']
    factor_daily['T5-3'] = factor_daily['DGS5']-factor_daily['DGS3MO']
    del factor_daily['DGS5']
    
    sp500['DATE'] = pd.to_datetime(sp500['DATE'].astype(str))
    factors_daily = pd.merge(factor_daily.iloc[:,[0,1,2,3,4,8,9,10,11]], 
                    sp500[sp500['DATE']>=factor_daily['DATE'][0]], 
                    on = 'DATE', how = 'outer',sort = True).dropna().reset_index(drop = True)
    factors_daily['e2c'] = (1-0.3)*(4/9)*0.5/(factors_daily['Adj Close'] + 0.5)*(factors_daily['VIXCLS']**2)
    del factors_daily['Adj Close']
#    factors_daily = pd.merge(factors_daily, fama.iloc[:,[0,1,2,3,5]], on = 'DATE', 
#                             how = 'left', sort = True)
#    factors_daily = pd.merge(factors_daily, vwretd, on = 'DATE', how = 'left', 
#                             sort = True).dropna()
    factors_daily = multimerge([factors_daily,fama.iloc[:,[0,1,2,3,5]],vwretd, 
                                swap], on = 'DATE', how = 'left').fillna(method = 'ffill').dropna().sort_values('DATE')
    factors_daily['DGS10^2'] = factors_daily['DGS10']**2 ##Capture convexity
    
    '''2. Weekly frequency'''
    factor_weekly = multimerge([lever1, lever2, stressind], on = 'DATE', how = 'outer')
    factor_weekly['DATE'] = pd.to_datetime(factor_weekly['DATE'].astype(str))
    
    '''3. Monthly frequency (CAN'T USE because release schedule is unsure)'''
    factor_monthly = pd.merge(sentiment, leading, on = 'DATE', how = 'outer', sort = True)
    factor_monthly['DATE'] = pd.to_datetime(factor_monthly['DATE'].astype(str))
    factors_monthly = pd.merge(factor_monthly, mutual.iloc[:,[0,3,4]], on = 'DATE', sort = True, how = 'outer')
    
    
    '''Delay data'''
    start = pd.to_datetime('1996-12-31')
    end = pd.to_datetime('2018-12-31')
    trad = pd.DataFrame(trading)
    trad = trad[trad['Date'] >= start]

    '''1. Delay daily data (delay by 1 day)'''
    factors_dd = origin.Time_delay(factors_daily, freq = 'D', n = 1, 
                        datecol = 'DATE', tradedates = trad).freq_delay()

    '''2. Delay weekly data (delay till next Thursday)'''
    factors_wd = origin.Time_delay(factor_weekly, freq = 'W', n = 4, 
                        datecol = 'DATE', tradedates = trad).freq_delay()

    '''3. Delay monthly data'''
    factor_m = mutual.copy()
    factor_m['mutual'] = factor_m.iloc[:,2]/factor_m.iloc[:,1]
    #factor_m = origin.Time_delay(factor_m, freq = 'M', n = 1, datecol = 'DATE',
    #                             tradedates = trad, end = '2018-12-31').freq_delay()
    from datetime import timedelta
    factor_m['DATE'] = factor_m['DATE'] + timedelta(30) 
    
    ##to much delay
    
    '''Merge non-delayed data (explanatory model)'''
    factors_org = pd.merge(factors_daily, factor_weekly, on = 'DATE', how = 'left', 
                       sort = True).fillna(method = 'ffill')
    
    '''Merge delayed factors data (predictive model)'''
    ##daily
#    factors = pd.merge(factors_dd, factors_wd, on = 'DATE', how = 'left', 
#                       sort = True).fillna(method = 'ffill')    
    ##weekly
    factors = pd.merge(factors_dd, factors_wd, on = 'DATE', how = 'right', 
                       sort = True).fillna(method = 'ffill')
    factors = pd.merge(factors, factor_m.loc[:,['DATE', 'mutual']], on = 'DATE', 
                       how = 'outer', sort =True).fillna(method = 'ffill').dropna()
    factors = pd.merge(pd.DataFrame(factors_wd.iloc[:,0]), factors, on = 'DATE', how = 'left', 
                       sort = True)
    
    '''Merge spread and factor data'''
    full_org = pd.merge(spread_f.iloc[:,[0,3]], factors_org, how = 'right', 
                on = 'DATE', sort = True).fillna(method = 'ffill').reset_index(drop = True)
    
    full = pd.merge(spread_f.iloc[:,[0,3]], factors, how = 'right', on = 'DATE', 
                    sort = True).fillna(method = 'ffill').reset_index(drop = True)
    #full['t10^2'] = full['DGS10^2']
    
    ''''diff-on-dff'''
    full.iloc[:,1:10] = full.iloc[:,1:10]-full.iloc[:,1:10].shift()
    full.iloc[:,21:24] = full.iloc[:,21:24]-full.iloc[:,21:24].shift()
    #full.columns = ['DATE', 'spread'] + ['Δ' + c for c in full.columns[2:]]
    full['DGS10^2'] = full['DGS10']**2
    full['lagged spread'] = full['spread'].shift()
    full = full.dropna()
    
#    del full['ΔTEDRATE']
#    del full['ΔNFCINONFINLEVERAGE']
#    del full['ΔSTLFSI']
#    #del full['ΔVIXCLS']
#    del full['Δumd']
    
    '''Take the difference'''
    ##factor 1:  DGS10;  factor 2: T10-T2
    ##factor 3:  VIX;    factor 4: TEDRate-3month
    ##factor 5:  TED-1y; factor 6: SP500
    ##factor 7:  mktrf;  factor 8: SMB
    ##factor 9:  HML;    factor 10: UMD
    ##factor 11: vwretd; factor 12: DGS10^2
    ##factor 13: lever1; factor 14: lever2
    ##factor 15: stress; factor 16: lagged spread 
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    
    factorcol = np.array([5,7,8,9,16])
    cols = np.array([1,2,4,5,9,10,13,14,18,20,22,23])
    r2 = 0
    loop = 0
    allr2 = {}
    allf = {}
    for r in range(6,8):
        for c in list(itertools.combinations(cols, r)):
            factorcol = np.array(c)
            #factorcol = np.append(factorcol, np.array([12])) #add the lagged spread
    #            full_orgdiff = full_org.copy()
    #            full_orgdiff.iloc[:,1:] = full_orgdiff.iloc[:,1:] - full_orgdiff.iloc[:,1:].shift()
    #            full_orgdiff = full_orgdiff.dropna()
    #            
    #            
    #            full_diff = full.copy()
    #            full_diff.iloc[:,factorcol+1] = full_diff.iloc[:,factorcol+1] - \
    #                                        full_diff.iloc[:,factorcol+1].shift()
    #            full_diff = full_diff.dropna()
            
    #        '''Normalize the data (optional)'''
    #            from sklearn.preprocessing import MinMaxScaler
    #            
    #            scaler = MinMaxScaler()
    #            full.iloc[:,2:18] = scaler.fit_transform(full.iloc[:,2:18])
            
        #    full_c = full.iloc[:,1:]-full.iloc[:,1:].shift()
        #    full_c = np.array(full_c.dropna())
        #    full_c[:,1:] = scaler.fit_transform(full_c[:,1:])
            
            '''Train/test split'''
            fullarr = np.array(full.iloc[:,1:])
            split = len(full)-200 #leave 1000 set of data for testing
            x_train = fullarr[:split,factorcol]#x_data[:split+1]
            y_train = fullarr[:split,0]#.loc[:split,'spread']
            
            x_test = fullarr[split:,factorcol]#x_data[split:]
            y_test = fullarr[split:,0]#.loc[split:, 'spread']
            
            
            '''Machine learning'''
            '''Simple Regression'''
            sm = LinearRegression().fit(x_train, y_train)
            score_sm = sm.score(x_test, y_test)
            train_sm = sm.score(x_train, y_train)
    #        '''Lasso Regression'''
    #        lasso = Lasso().fit(x_train, y_train)
    #        score_ls = lasso.score(x_test, y_test)
            
            
            '''Gradient Boosted'''
            clf = GradientBoostingRegressor().fit(x_train, y_train)   
            score_gb = clf.score(x_test, y_test)
            train_gb = clf.score(x_train, y_train)
            
            '''Random Forest'''
            clf2 = RandomForestRegressor().fit(x_train, y_train)
            score_rf = clf2.score(x_test, y_test)
            train_rf = clf2.score(x_train, y_train)
            
            '''Neutral Network'''
            '''Multi-Layer Perception'''
            nnclf =  MLPRegressor(hidden_layer_sizes = [10], solver='lbfgs',
                                 random_state = 0).fit(x_train, y_train)
            score_n = nnclf.score(x_test, y_test)
            train_n = nnclf.score(x_train, y_train)
#            
#            [score_sm, score_gb, score_rf, score_n]
#            '''PCA'''
#            from sklearn.decomposition import PCA
#            pca = PCA(n_components = 6).fit(full.iloc[:,2:])
            allr2[str(c)] = [score_sm, score_gb, score_rf, score_n]
            allf[str(c)] = full.columns[factorcol+1].tolist()
            
    #            loop += 1
            if max([score_sm, score_gb, score_rf, score_n]) > r2:
                r2 = max([score_sm, score_gb, score_rf, score_n])
                features = full.columns[factorcol+1].tolist()
#            plt.plot(full.iloc[:,[1,10]])
#            plt.legend(['spread','sp500'])
    '''Summary of R2'''
    #[full.columns[factorcol+1].tolist(),[score_sm, score_ls, score_gb, score_rf, score_n]]
 
    '''Save to excel'''
    writer = pd.ExcelWriter('combinations_weekly.xlsx')
    pd.DataFrame.from_dict(allf, orient='index').to_excel(writer, 'features')
    pd.DataFrame.from_dict(allr2, orient='index').to_excel(writer, 'r2')
    writer.save()
    
    
    '''Plot ACF and PACF'''
    import statsmodels.tsa.stattools
    pacf = statsmodels.tsa.stattools.pacf(full.iloc[:,1])
    plt.figure()
    plt.bar(range(0,len(pacf)),pacf)
    plt.title('PACF')
    
    acf = statsmodels.tsa.stattools.acf(full.iloc[:,1])
    plt.figure()
    plt.bar(range(0,len(acf)),acf)
    plt.title('ACF')
    
    y_pred2 = []
    for r in range(split,len(full)):
        x_train = fullarr[:split,factorcol]
        y_train = fullarr[:split,0]
        
        x_test = fullarr[r,factorcol]#x_data[split:]
#        y_test = fullarr[r:,0]

        sm = LinearRegression().fit(x_train, y_train)
        y_pred2.append(sm.intercept_ + np.sum(sm.coef_*x_test))
    
    spr = pd.DataFrame(full['spread'][222:])
    
    yy = pd.DataFrame(full.iloc[:,1])
    yy['pred'] = list(nnclf.predict(x_train)) + list(nnclf.predict(x_test))
    
    yy['spread'] = yy['spread'] + spr['spread'].shift()
    yy['pred'] = yy['pred'] + spr['spread'].shift()
    
    plt.figure()
    plt.plot(full['DATE'], yy)
    plt.legend(['actual','model'])
    plt.title('Multilayer Perception - Spread')
    
    from sklearn.metrics import r2_score
#----------------------------old version---------------------------------#
#def checkdate(d):
#    if (d.month == 1) and (d.day == 1): ##New Year
#        return False
#    elif (d.month == 1) and (d.weekday() == 0) and ((d.day >= 15) or (d.day <= 21)):
#        return False #Martin Luther Day
#    elif (d.month == 2) and (d.weekday() == 0) and ((d.day >= 15) or (d.day <= 21)):
#        return False #President day
#    elif (d.month == 4) and (d.weekday() == 4) and 
#        return False #Good Friday
#    elif (d.month == 5) and (d.weekday() == 0) and ((d.day >= 25) or (d.day <= 31)):
#        return False #Memorial day
#    elif (d.month == 7) and (d.day == 4):
#        return False #Independence day
#    elif (d.month == 9) and (d.weekday() == 0) and ((d.day >= 1) or (d.day <= 7)):
#        return False #Labor day
#    elif (d.month == 11) and (d.weekday() == 3) and ((d.day >= 22) or (d.day <= 28)):
#        return False #Thanksgiving
#    elif (d.month == 12) and (d.day == 25):
#        return False
#    else:
#        return True