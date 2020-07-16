from pandas_datareader import data as pdr
import quantstats as qs
from scipy.stats import norm 
import yfinance as yf
from tqdm import tqdm
import math
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np;np.random.seed(0)
import pandas as pd
import statsmodels.api as sm
import seaborn as sns; sns.set()
from scipy import stats
from hmmlearn import hmm
from sklearn.decomposition import PCA
import regimeDetection
import strategies
import utilityFuncs
import os
import regimeDetection as rgd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_stats(portfolioValue,rebalancing):
    dataM=portfolioValue.Return
    stocks = ["SPY"]
    start = datetime(2015,4,1)
    end = datetime(2020,6,1)
    factorData = pdr.get_data_yahoo(stocks, start=start, end=end)["Adj Close"].pct_change()
    rfr=pdr.get_data_yahoo(["^IRX"], start=start, end=end)["Adj Close"]/36500
    together=factorData.join(dataM).dropna()
    together=together.join(rfr)
    correl1= round(np.corrcoef(together.SPY,together.Return)[0][1],2)
    print ("Correlation to SP500: ", correl1)
    print ("Kurtosis:",round(stats.kurtosis(together.Return),2))
    print ("Skewness:",round(stats.skew(together.Return),2))
    print ("Volatility:",round(np.std(together.Return)*math.sqrt(252)*100,2))
    sharpeRatio= np.mean(together.Return-together["^IRX"])/np.std(together.Return-together["^IRX"])
    print ("Sharpe Ratio:",round(sharpeRatio*math.sqrt(252),3))
    mdd=qs.stats.max_drawdown(dataM)
    cagr=qs.stats.cagr(dataM)
    print ("Max Drawdown in %:", round(mdd*100,3))
    print ("CAGR in %:", round(cagr*100,2))
    print ("Sortino Ratio: ",round(qs.stats.sortino(together.Return),2))
    
    #VaR Calculation
    
    h = 1. # horizon of 10 days
    mu_h = np.mean(portfolioValue.Return) # this is the mean of % returns over 10 days - 10%
    sig = np.std(portfolioValue.Return)*np.sqrt(252)  # this is the vol of returns over a year - 30%
    sig_h = sig * np.sqrt(h/252) # this is the vol over the horizon
    alpha = 0.01

    VaR_n = norm.ppf(1-alpha)*sig_h - mu_h 
    valueV=portfolioValue.iloc[-1].Value_CAD*VaR_n
    print("99% 1 day VaR :", round(VaR_n*100,2),"% or", round(valueV),"$")

    CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_h - mu_h
    valueV=portfolioValue.iloc[-1].Value_CAD*CVaR_n
    print("99% 1 day CVaR/ES :", round(CVaR_n*100,2),"% or", round(valueV),"$")

    h = 10. # horizon of 10 days
    sig_h = sig * np.sqrt(h/252)
    VaR_n = norm.ppf(1-alpha)*sig_h - mu_h 
    valueV=portfolioValue.iloc[-1].Value_CAD*VaR_n
    print("99% 10 day VaR :", round(VaR_n*100,2),"% or", round(valueV),"$")

    CVaR_n = alpha**-1 * norm.pdf(norm.ppf(alpha))*sig_h - mu_h
    valueV=portfolioValue.iloc[-1].Value_CAD*CVaR_n
    print("99% 10 day CVaR/ES :", round(CVaR_n*100,2),"% or", round(valueV),"$")
    
    
    #Period Wise Return Calcualtion
    print ()
    print ("Returns -Period Wise")
    returnsforTWR=[]
    
    for i in range(len(rebalancing)-1):
       idx1= (portfolioValue.index.get_loc(rebalancing[i]))
       idx2= (portfolioValue.index.get_loc(rebalancing[i+1]))
       if i!=0:
           irr=round( 100*((portfolioValue.Value_CAD.iloc[idx2-1]/portfolioValue.Value_CAD.iloc[idx1-1])-1),2)
           print ("Period",i+1,":", irr)
          
       else:
           irr=round(100*((portfolioValue.Value_CAD.iloc[idx2-1]/portfolioValue.Value_CAD.iloc[idx1])-1),2)
           print ("Period",i+1,":", irr)

       returnsforTWR.append(irr/100)

    print ("Time Weighted Return:", (np.prod(np.add(returnsforTWR,1))-1)*100)

    
def portfolioGraphsandStats(returns):
    qs.reports.full(returns)


def usdcadExposures(portfolioValue):
    
    labels = ["CAD ", "USD"]
    y = np.vstack([portfolioValue["CADTickers"]/portfolioValue["Value_CAD"], (1-(portfolioValue["CADTickers"]/portfolioValue["Value_CAD"]))])
    plt.stackplot(portfolioValue.index, y,labels=labels)
    plt.title("USD/CAD Exposure")
    plt.legend()


def weightsEvolution(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD):
    
    labels = ["Equity ", "Credit" , "Alternatives","Cash","Gold","FI"]
    equityValue=((portfolioValue[tickerEquityCAD].sum(axis=1))+((portfolioValue[tickerEquity].multiply(portfolioValue['Adj Close'],axis=0).sum(axis=1)))).divide(portfolioValue["Value_CAD"])
    creditValue=((portfolioValue[tickerCreditCAD].sum(axis=1))+((portfolioValue[tickerCredit].multiply(portfolioValue['Adj Close'],axis=0).sum(axis=1)))).divide(portfolioValue["Value_CAD"])
    altsValue=((portfolioValue[tickerAltsCAD].sum(axis=1))+((portfolioValue[tickerAlts].multiply(portfolioValue['Adj Close'],axis=0).sum(axis=1)))).divide(portfolioValue["Value_CAD"])
    cashValue=(portfolioValue["Cash"]).divide(portfolioValue["Value_CAD"])
    goldValue=(portfolioValue["CGL.TO"]).divide(portfolioValue["Value_CAD"])
    fiValue=((portfolioValue["IEF"].multiply(portfolioValue['Adj Close'],axis=0))).divide(portfolioValue["Value_CAD"])
    fig = plt.figure()
    ax = plt.subplot(111)
    
    y=np.vstack([equityValue,creditValue,altsValue,cashValue,goldValue,fiValue])
    plt.stackplot(portfolioValue.index, y,labels=labels)
    plt.title("Asset Class Exposure")
    ax.legend(loc='right', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
    plt.show()
    
def nvCalculator(portfolioValue,selectedIndex,equityTickers,creditTickers,altsTickers,tickerEqNamesUS,tickerEqNamesCAD,tickerCreditNamesUSD,tickerCreditNamesCAD,tickerAltsNamesUSD,tickerAltsNamesCAD):
    print ("##################Exposures on any day#########################")
    print ("Date",portfolioValue.index[selectedIndex].date())
    print ("Portfolio Value : ", round(portfolioValue["Value_CAD"].iloc[selectedIndex]))
    
    nvEquity=portfolioValue[equityTickers].iloc[selectedIndex]
    nvEquity.index=tickerEqNamesUS+tickerEqNamesCAD
    print ()
    print ("Equity Exposure : ",sum(round(nvEquity)))
    utilityFuncs.goodPrint(round(nvEquity).to_string())
    
    nvCredit=portfolioValue[creditTickers].iloc[selectedIndex]
    nvCredit.index=tickerCreditNamesUSD+tickerCreditNamesCAD
    print ()
    print ("Credit Exposure : ",sum(round(nvCredit)))
    utilityFuncs.goodPrint(round(nvCredit).to_string())
    
    nvAlts=portfolioValue[altsTickers].iloc[selectedIndex]
    nvAlts.index=tickerAltsNamesUSD+tickerAltsNamesCAD
    print ()
    print ("Alternatives Exposure : ",sum(round(nvAlts)))
    utilityFuncs.goodPrint(round(nvAlts).to_string())
    
    print ()
    nvGold=portfolioValue["CGL.TO"].iloc[selectedIndex]
    print ("Gold_CAD Exposure : ", round(nvGold))
    
    nvUST=portfolioValue["IEF"].iloc[selectedIndex]
    print ("US Treasury Exposure : ", round(nvUST))
    
    nvCash=portfolioValue["Cash"].iloc[selectedIndex]
    print ("Cash : ", round(nvCash))

def benchmarkComp(portfolioValue):
    
    benchmarkData=pd.read_csv("Data/Benchmark.csv",index_col=0,parse_dates=True).sort_index().loc["03-2015":"05-2020"]     
    benchmarkData.drop(["Name","Code","Return"],axis=1,inplace=True)
    portmonthReturns=[]
    
    portmonthValue=[100000]
    for i in range(2015,2021):
        for j in range(1,13):
            try:
                dataMonthly= (portfolioValue.loc[str(j)+"-"+str(i)].Value_CAD)
                returnMonthly=(dataMonthly.iloc[-1]/dataMonthly.iloc[0])-1
                portmonthReturns.append(round(returnMonthly,4))
                portmonthValue.append(dataMonthly.iloc[-1])
            except:
                pass
            
    portmonthReturns= portmonthReturns[:-1]
    portmonthValue= portmonthValue[:-1]
    benchmarkData["Portfolio"]=portmonthValue
    benchmarkData["Port_Returns"]=benchmarkData["Portfolio"].pct_change()
    benchmarkData["Bench_Returns"]=benchmarkData["Index"].pct_change()
    
    
    for i in range(2015,2021):
        for j in [4,10]:
            if (j==4 and i==2015) or (j==10 and i==2020):
                continue
            else:
                idx= (benchmarkData.loc[str(j)+"-"+str(i)].index[0])
                idx1= (benchmarkData.index.get_loc(idx))
                newReturn=((benchmarkData.iloc[idx1].Portfolio)/(benchmarkData.iloc[idx1-1].Portfolio+10000))-1
                benchmarkData.loc[idx].Port_Returns=newReturn
    
    
    plt.plot(benchmarkData["Port_Returns"].cumsum()*100,label="Portfolio")
    plt.plot(benchmarkData["Bench_Returns"].cumsum()*100,label="Benchmark")
    plt.xlabel("Years")
    plt.ylabel("Returns")
    plt.legend()
    plt.title("Performance of Portfolio vs Benchmark")
    plt.show()
    trackingError=np.std(benchmarkData["Port_Returns"]-benchmarkData["Bench_Returns"])*np.sqrt(12)
    informationRatio=  (qs.stats.cagr(benchmarkData["Port_Returns"]) - qs.stats.cagr(benchmarkData["Bench_Returns"])) /trackingError
    excessive_return=benchmarkData.Port_Returns-benchmarkData.Bench_Returns
    
    print ("Tracking Error: ", round(trackingError,3))
    print ("Information Ratio: ", round(informationRatio,3))
    stocks = ["SPY"]
    start = datetime(2015,3,1)
    end = datetime(2020,5,31)
    factorData = pdr.get_data_yahoo(stocks, start=start, end=end)["Adj Close"]
    monthlySP=factorData.resample("m").last()
    monthlySP=monthlySP.pct_change().dropna()
    
    (beta, alpha) = stats.linregress(list(monthlySP.SPY),list(benchmarkData.Port_Returns.dropna()))[0:2]
    print ("Beta:", round(beta,2))
    print ("Alpha:", round(alpha*12*100,3))
    
    return benchmarkData

def txnCostCalc(portfolioValue,rebalancing):
    
    txnCost=0
    for i in range(len(rebalancing)):
        rebalDate=rebalancing[i]
        if i==0:
            txnCost=(portfolioValue.loc[rebalDate].Value_CAD)*0.001
            
        else:
            idx1= (portfolioValue.index.get_loc(rebalancing[i]))
            idx2=idx1-1
            a= portfolioValue.iloc[idx1]
            b= portfolioValue.iloc[idx2]
            txnCost+= (sum((abs(a-b)[portfolioValue.columns[:25]]))*0.001)
            

    print ("Transaction Costs: ",round(txnCost,2),"$")    



def getExposure(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD,tickerHedge,tickerHedgeCAD,date='2020-06-01'):
    w = portfolioValue.loc[date][:-7]/(portfolioValue.loc[date][:-7].sum())
    EQw = w[tickerEquity].sum()
    CRw = w[tickerCredit].sum()
    Alt_w = w[tickerAlts].sum()
    Hedge_w = w[tickerHedge].sum()
    EQw_CAD = w[tickerEquityCAD].sum()
    CRw_CAD = w[tickerCreditCAD].sum()
    Alt_w_CAD = w[tickerAltsCAD].sum()
    Hedge_w_CAD = w[tickerHedgeCAD].sum()
    cash = w['Cash']
    # list of strings
    lst = [EQw, CRw, Alt_w, Hedge_w, EQw_CAD, CRw_CAD, Alt_w_CAD, Hedge_w_CAD, cash]
    df = pd.DataFrame(lst, index =['EQ_USD', 'CR_USD', 'Alt_USD', 'Hedge_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD', 'Hedge_CAD', 'Cash'], columns =['Weight'])
    return df


# Return Attribution
def getReturn(PortfolioValue,rebalancing,date='2020-06-01'):
    PortfolioValue = PortfolioValue.copy()
    PortfolioValue['Value']=PortfolioValue.sum(axis=1)
    PortfolioValue['Return_temp']=PortfolioValue['Value'].pct_change().dropna()
    PortfolioValue.loc[list(PortfolioValue.loc[PortfolioValue.index.isin(rebalancing)][1:].index),'Return_temp']=np.nan
    riskReturns = PortfolioValue['Return_temp'].fillna(method='ffill')
    return riskReturns.loc[:date]


    
def getReturnAttribution(portfolioValue,rebalancing,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD):
    portValue=portfolioValue.copy()
    portValue['regime']=portfolioValue[['Cash','IEF','CGL.TO']].sum(axis=1)
    portValue.drop(['Cash','IEF','CGL.TO'],axis=1,inplace=True)
    rb=list(rebalancing)
    rb.append(pd.to_datetime('2020-07-03'))
    tickerAll=tickerEquity+tickerCredit+tickerAlts+tickerEquityCAD+tickerCreditCAD+tickerAltsCAD
    tickerAll.append('regime')
    moneyAdd=[(portValue.loc[rb[i]:rb[i+1],tickerAll].iloc[-2]-portValue.loc[rb[i]:rb[i+1],tickerAll].iloc[0]) for i in range(len(rb)-1)]
    moneyAdd=pd.DataFrame(moneyAdd)
    returnAttr=pd.Series()
    returnAttr['US Equity']=moneyAdd[tickerEquity].sum().sum()
    returnAttr['CAD Equity']=moneyAdd[tickerEquityCAD].sum().sum()
    returnAttr['US Credit']=moneyAdd[tickerCredit].sum().sum()
    returnAttr['CAD Credit']=moneyAdd[tickerCreditCAD].sum().sum()
    returnAttr['US Alternative']=moneyAdd[tickerAlts].sum().sum()
    returnAttr['CAD Alternative']=moneyAdd[tickerAltsCAD].sum().sum()
    returnAttr['Regime']=moneyAdd['regime'].sum()
    returnAttr["Return Attribution"]=returnAttr*(portValue.Value_CAD[-1]-200000)/returnAttr.sum()
    return returnAttr["Return Attribution"]

def getRiskAttribution(portfolioValue,rtnBreakDown,rtnBreakDownCAD,w,date='2020-06-01'):
    returns = [rtnBreakDown[0], rtnBreakDown[1], rtnBreakDown[2], rtnBreakDownCAD[0], rtnBreakDownCAD[1], rtnBreakDownCAD[2]]
    name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
    w = w.loc[name]/w.loc[name].sum()
    df = pd.DataFrame(returns).T.dropna()
    df.columns = name
    Q = df.iloc[-1000:].cov()
    riskAttribution = np.dot(np.array(w.T),np.array(Q))
    risk = pd.DataFrame(riskAttribution,columns= name,index = ['Risk Attribution'])
    riskAttr = risk/risk.sum(axis=1)[0]
    riskAttr=riskAttr.T
    riskAttr=riskAttr*w.loc[riskAttr.index].values
    riskAttr=riskAttr/riskAttr.sum()
    riskAttr.index=  ['US Equity', 'US Credit', 'US Alternative', 'CAD Equity', 'CAD Credit', 'CAD Alternative']
    return riskAttr

