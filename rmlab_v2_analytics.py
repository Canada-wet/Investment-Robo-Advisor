'''
@Authors : Pulkit Gaur, Hongyi Wu, Jiaqi Feng

This code is written for the course MMF 2025 - Risk Management Laboratory
by Dr. Dan Rosen. 
United All Weather Robo Advisor
'''
########################################################################

## Import Statements, please install hmmlearn & quantstats
    
from pandas_datareader import data as pdr
import quantstats as qs
from scipy.stats import norm 
import yfinance as yf
import fix_yahoo_finance as yf
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
import metricsCalculator
import regimeDetection as rgd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import riskModel
os.getcwd()

#######################################################################

"""## Initial Data Pre Processing - Exploration"""

stocks = ["SCO","SPY","GLD","VWO","IEF","EMB","lqd","VNQ","MNA","CAD=X","^IRX"]
start = datetime(2010,1,1)
end = datetime(2020,6,1)

data = pdr.get_data_yahoo(stocks, start=start, end=end)
data = data["Adj Close"]
rf = data.iloc[1:,-1]/252
cad=data["CAD=X"]
data = data.iloc[1:,:-2]
returns=data.pct_change().dropna()
clmns='Oil,SPX,Gold,EM EQ,US Bond,EMD,US HY,REIT,Hedge Fund'.split(',')
dataIdx=data.index.values
dataNamed=pd.DataFrame(data.values,index=dataIdx,columns=clmns).dropna()
rtnNamed=dataNamed.pct_change().dropna()

#######################################################################

"""## Actual Data and Portfolio Construction"""

#Portfolio Construction
#US Equity Tickers
tickerEquity=['XLY','XLI','XLF','XLV','XLK','XLP']
tickerEqNamesUS=["Consumer Discretionary", "Industrial", "Financial", "Health Care","Technology","Consumer Staples"]

#CAD Equity Tickers
tickerEquityCAD=['XMD.TO','XFN.TO','ZUH.TO','XIT.TO','ZDJ.TO']
tickerEqNamesCAD=["Mid_Small_CAD", "Financial_CAD", "Health Care_CAD", "Information Technology_CAD", "DJI_CAD"]

#US Credit Tickers
tickerCredit=["EMB","HYG",'LQD','MBB']
tickerCreditNamesUSD= [ "Emerging Markets", "High Yield", "Investment Grade", "Mortgage Backed Securities"]

#CAD Credit Tickers
tickerCreditCAD=['ZEF.TO','XHY.TO','ZCS.TO','XQB.TO']
tickerCreditNamesCAD= [ "Emerging Markets_CAD", "High Yield_CAD", "Corporate Bonds_CAD","Investment Grade_CAD"]

#Hedge Assets- gold in CAD and us treasury- used when regime changes
tickerHedge=['IEF']
tickerHNamesUSD=["US_Treasury"]
tickerHedgeCAD=['CGL.TO']
tickerHNamesCAD=["Gold_CAD"]

#US Alternatives Tickers
tickerAlts=['PSP','IGF','VNQ','MNA']
tickerAltsNamesUSD=["PE", "Infra", "REITs", "HF"]

#CAD Alternatives Tickers
tickerAltsCAD=['CGR.TO','CIF.TO']
tickerAltsNamesCAD=["REITs_CAD", "Infra_CAD"]

#Downloading the FX and Overnight Interest Rate Data
start = datetime(2015,4,1)

end = datetime(2020,6,1)
fx = pdr.get_data_yahoo("CAD=X", start=start, end=end)
fxData = fx["Adj Close"]
oRates=pd.read_csv("Data/canadaOvernight.csv",index_col=0,parse_dates=True).sort_index()

#Downloading data from Yahoo Finance API
stocks = tickerEquity+tickerCredit+tickerAlts+tickerHedge+["SPY","CAD=X","^IRX"]
stocksCAD = tickerEquityCAD+tickerCreditCAD+tickerAltsCAD+tickerHedgeCAD+["SPY","CAD=X","^IRX"]
start = datetime(2010,1,1)
end = datetime(2020,6,1)
price,rtn=utilityFuncs.pull_data(stocks)
priceCAD,rtnCAD=utilityFuncs.pull_data(stocksCAD)
commonDate=[i for i in price.index if i in priceCAD.index]
priceMerged=pd.concat([price.loc[commonDate],priceCAD.loc[commonDate]],axis=1)

start = datetime(2015,4,1)
end = datetime(2020,6,1)
priceHedge= pdr.get_data_yahoo(tickerHedge+tickerHedgeCAD, start=start, end=end)["Adj Close"]
priceHedge= priceHedge.ffill(axis=0).dropna()

#Calculate weights for US tickers
rtnTotal,nvTotal,wTotal,rtnBreakDown=utilityFuncs.make_port(price,tickerEquity,tickerCredit,tickerAlts,True)
#Calculate weights for CAD tickers
rtnTotalCAD,nvTotalCAD,wTotalCAD,rtnBreakDownCAD=utilityFuncs.make_port(priceCAD,tickerEquityCAD,tickerCreditCAD,tickerAltsCAD,True)

#Merge the Weights in one file
mutualDate=[i for i in wTotal.index if i in wTotalCAD.index]
weightMerged=pd.concat([wTotal.loc[mutualDate]*0.556,wTotalCAD.loc[mutualDate]*0.444],axis=1)
weightMerged.to_pickle('weights.pkl')

######################################################################
    
# Regime Detection and Signals Generation

if 'Signal.pkl' in os.listdir(os.getcwd()+'\\Data'):
    
    signalSeries=pd.read_pickle('Data\\Signal.pkl')
    
else:
    
    #Using different factors like Fixed Income Vol, Equity Vol, FX Vol, Term Premium,
    #Credit Premium, Move Index etc.
    dataHMM=pd.read_excel('Data\\HMM_data.xlsx',index_col=0)
    start = datetime(2008,1,1)
    end = datetime(2020,5,31)
    
    term_premium = pdr.get_data_yahoo(['^TYX','^IRX'], start=start, end=end)
    term_premium = term_premium["Adj Close"]
    term_premium = term_premium['^TYX']-term_premium['^IRX']
    
    dataHMM=dataHMM.loc[term_premium.index]
    dataHMM.iloc[:,-1]=term_premium.values
    
    dataInput=dataHMM
    dataInput_m=dataInput.resample('m').last()
    dataNormed1=rgd.percentile_data(dataInput,1)
    
    EMIndex1=(dataNormed1*[0.2,0.2,0.2,0.2,0.15,0.05]).sum(axis=1)# 1-year version
    
    model=hmm.GMMHMM(n_components=3, covariance_type="full",random_state= 0)
    
    newStates1=[]
    
    #Learning from the data
    for i in tqdm(range(251,EMIndex1.size+1)):
        
        dataHMMTemp=EMIndex1.iloc[:i].values.reshape(-1,1)
        states=rgd.fix_states(model.fit(dataHMMTemp).predict(dataHMMTemp),EMIndex1.iloc[:i].values)
        newStates1.append(states[-1])
    
    dataHMMInit1=EMIndex1.iloc[:250].values.reshape(-1,1)
    modelInit1=model.fit(dataHMMInit1)
    stateInit1=rgd.fix_states(modelInit1.predict(dataHMMInit1),dataHMMInit1)
    updatedStates1=pd.Series(list(stateInit1)+newStates1,index=EMIndex1.index)
    signalOff=[i for i in range(1,updatedStates1.size) if updatedStates1[i-1]==1 and updatedStates1[i]==2]
    signalOn=[i for i in range(1,updatedStates1.size) if updatedStates1[i-1]==2 and updatedStates1[i]==1]

    #Generating Signal Series
    signalSeries=pd.Series(0,index=updatedStates1.index)
    signalSeries[signalOn]=1
    signalSeries[signalOff]=-1
    signalSeries.to_pickle('Data\\Signal.pkl')
    
#######################################################################

'''Rebalancing and Portfolio Allocation'''

    
#Renaming the weights
weightsAll=weightMerged

#Finding the dates to rebalance    
myMask=[]
temp=[]
x=2015

for i in range(6):
  temp.append(date(x+i,4,1))
  temp.append(date(x+i,10,1))

trialList=list(temp)
rebalancing=[]

#Getting all the rebalancing dates
for i in trialList:
  try:
    a= (weightsAll.loc[i])
    rebalancing.append(i)
  except:
    try:
       a= (weightsAll.loc[i+timedelta(days=1)])
       rebalancing.append(i+timedelta(days=1))
    except:
       try:
            a=(weightsAll.loc[i+timedelta(days=2)])
            rebalancing.append(i+timedelta(days=2))
       except:
            try:
                a=(weightsAll.loc[i+timedelta(days=3)])
                rebalancing.append(i+timedelta(days=3))
            except:
                   pass
               
for i in list(weightsAll.index):
    i = i.to_pydatetime().date()
    if i in (rebalancing):
        myMask.append(True)
    else:
        myMask.append(False)


#Using the mask, this dataframe contains all the portfolio weights
ERCWeight=weightsAll.loc[myMask]

#Automated Backtesting for Main Portfolio

#This is the value that is used for investing in main portfolio
start=90000
portfolioValue=priceMerged.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
portfolioValue= (portfolioValue[ERCWeight.columns])
price=priceMerged[ERCWeight.columns].dropna()
price=price.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()

#This two lists will hold the principal amount and cash available.
investment=[]
cash=[]

#Backtesting Code
for i in range(len(ERCWeight)):
    
  rebalanceDate=ERCWeight.index[i]

  #finding start and end date for a rebalancing period.  
  try:
    endDate=ERCWeight.index[i+1] - timedelta(days=1)
  except:
    endDate=date(2020,6,1)

  relevantData=portfolioValue[rebalanceDate:endDate]
  rebalanceDate=relevantData.index[0]
  endDate=relevantData.index[-1]
  
  
  #Money Allocated to each of the asset in CAD
  moneyAllocated=start*ERCWeight.iloc[i]
  
  #Finding FX rate on first day and converting the USD prices to CAD
  try:
      fxConvert=fxData.loc[rebalanceDate]
  except:
      fxConvert=fxData.loc[rebalanceDate.date()-timedelta(days=1)]
      
      
  usTickers=[i for i in list(price.columns) if (i[-2:] != "TO")]
  priceinCAD=price.copy().loc[rebalanceDate]
  priceinCAD[usTickers]*=fxConvert

  # Number of Units to buy for each asset in each period
  noofUnits=moneyAllocated.divide(priceinCAD)
  
  # Adding all the values and evolution of value for each asset in a period.
  portfolioValue[rebalanceDate:endDate]=portfolioValue[rebalanceDate:endDate]*list(noofUnits)
  investment.extend([100000+(i*10000)]*len(portfolioValue[rebalanceDate:endDate]))
  cash.extend([10000+(i*1000)]*len(portfolioValue[rebalanceDate:endDate]))

  #Figuring out the value of portfolio on the last day in CAD that will be used for
  #reinvesting next period
  priceinCAD=portfolioValue.copy().loc[endDate]
  
  try:
      fxConvert=fxData.loc[endDate]
  except:
      fxConvert=fxData.loc[endDate.date()-timedelta(days=1)]
  
  priceinCAD[usTickers]*=fxConvert
  endvalue=priceinCAD.sum()
  
  #This 9000 means the amount that is added in the next rebalancing period.
  start=9000+endvalue


#Adding the column for cash in the dataframe
portfolioValue["Cash"]=cash

#Adding the regime strategy overlay

trades=signalSeries.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
moneyAccount=portfolioValue.Cash.copy()
openPos=0

#This list has all the buy and sell dates for each round trip of trade.
regimeDates=[]

for i in range(len(moneyAccount)):
    
    try:
        currentIndex=moneyAccount.index[i]        
        if trades[currentIndex] == 1 and openPos==0:
            buyIndex=currentIndex
            buyPrice=priceHedge.loc[(currentIndex.date())]
            openPos=1

        elif trades[moneyAccount.index[i]] == -1 and openPos==1:
            sellPrice=priceHedge.loc[(currentIndex.date())]
            openPos=0
            regimeDates.append([buyIndex,currentIndex])
            
    except:
            pass

# This dataframe holds the price for GOLD and US Treasury
priceHedge2=priceHedge.copy()

for i in priceHedge.index:
    if i not in portfolioValue.index:
        priceHedge2.drop(i,inplace=True)
        
priceHedge=priceHedge2

#This holds the price for the assets during each round trip
tradeData=[]
for i in range(len(regimeDates)):   
    
    buyDate= regimeDates[i][0]
    sellDate= regimeDates[i][1]
    goldData=priceHedge.loc[buyDate:sellDate]["CGL.TO"]/priceHedge.loc[buyDate]["CGL.TO"]
    treaData=priceHedge.loc[buyDate:sellDate].IEF/priceHedge.loc[buyDate].IEF
    tradeData.append([goldData,treaData])

   
# Finding the value of cash, gold and US Treasury for the 5 years
# and M2m/PL calculation    
cashValue=[moneyAccount.iloc[0]]
treaValue=[0]
goldValue=[0] 
j=0   
buyDates= [i[0] for i in regimeDates]  
sellDates= [i[1] for i in regimeDates]   
numberofDays=0
openPos=False

for i in range(len(portfolioValue)-1):
    
    currentIndex=portfolioValue.index[i]
    ORate=oRates.loc[currentIndex.date()]/36500
    
    if currentIndex in rebalancing[1:]:
        
        if openPos==True:
            cashValue[i:i+numberofDays+1]=np.add(cashValue[i:i+numberofDays+1],1000)
        else:
            cashValue[i]=cashValue[i]+1000
    
    if openPos==True:
        
        if numberofDays>0:
            numberofDays-=1
            continue
        
        elif numberofDays==0:
            cashValue[i]=goldValue[i]+treaValue[i]+cashValue[i]
            goldValue[i]=0
            treaValue[i]=0
            openPos=False
           
         
    if currentIndex in buyDates:
        
        numberofDays=len(tradeData[j][0])-2
        goldData=np.multiply(list(tradeData[j][0]),float(cashValue[i]/2))
        treaData=np.multiply(list(tradeData[j][1]),float(cashValue[i]/2))      
        goldValue[i]=(cashValue[i]/2)
        treaValue[i]=(cashValue[i]/2)
        cashValue[i]=0
        goldValue.extend(list(goldData[1:]))
        treaValue.extend(list(treaData[1:]))
        cashValue.extend(len(goldData[1:])*[0])
        j+=1
        openPos=True

    else:
         
         cashValue.append((cashValue[i])*(1+float(ORate)))
         treaValue.append(0)
         goldValue.append(0)


#Adding the value of the regime assets in the main dataframe.
portfolioValue["Cash"]=cashValue
portfolioValue["CGL.TO"]=goldValue
portfolioValue["IEF"]=treaValue

#Adding some more columns for convinience in analysis
usTickers.append("IEF")
cadTickers=list(set(portfolioValue.columns)-set(usTickers))

portfolioValue["USDTickers"]=portfolioValue[usTickers].sum(axis=1)
portfolioValue["CADTickers"]=portfolioValue[cadTickers].sum(axis=1)

portfolioValue=portfolioValue.join(fxData)
portfolioValue.ffill(axis=0,inplace=True)
portfolioValue["USDTickers_CAD"]=portfolioValue["USDTickers"].multiply(portfolioValue["Adj Close"])
    
portfolioValue["Principal"]=investment
portfolioValue["Value_CAD"]=portfolioValue["CADTickers"]+portfolioValue["USDTickers_CAD"]

rebalancing = portfolioValue[~portfolioValue['Principal'].diff().isin([0])].index
portfolioValue["Return"]=portfolioValue["Value_CAD"].pct_change()
portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Return']=(portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])/((portfolioValue.shift(1).loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])+10000)-1

###################################

''' Analysis and Metrics Calculations'''
print ()
print ("###################Portfolio Stats##################")
equityTickers=tickerEquity+tickerEquityCAD
creditTickers=tickerCredit+tickerCreditCAD
altsTickers=tickerAlts+tickerAltsCAD

#Important Stats for Performance
metricsCalculator.get_stats(portfolioValue,rebalancing)

#Transaction Costs
metricsCalculator.txnCostCalc(portfolioValue,rebalancing)

#Benchmark Comparison
print ()
print ("###################Benchmark Comparison##################")
benchmarkData=metricsCalculator.benchmarkComp(portfolioValue)


#Generate graphs for the portfolio
returnData=portfolioValue.Return.dropna()
metricsCalculator.portfolioGraphsandStats(returnData)

plt.style.use("ggplot")
#Evolution of USD/CAD exposure
metricsCalculator.usdcadExposures(portfolioValue)
sns.set()


plt.style.use("ggplot")
#Asset Classes Weights Evolutions
metricsCalculator.weightsEvolution(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD)
sns.set()

#Notional Value in each Asset
metricsCalculator.nvCalculator(portfolioValue,len(portfolioValue)-1,equityTickers,creditTickers,altsTickers,tickerEqNamesUS,tickerEqNamesCAD,tickerCreditNamesUSD,tickerCreditNamesCAD,tickerAltsNamesUSD,tickerAltsNamesCAD)


plt.style.use("ggplot")
#Exposure Plots
exposure = metricsCalculator.getExposure(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD,tickerHedge,tickerHedgeCAD,'2020-06-01')
exposure['Weight'].plot.pie(autopct='%.2f', fontsize=15, figsize=(12, 12))
plt.title("Exposures",fontsize=25)
plt.show()
sns.set()

#Return Attribution
print ()
colors=["#376e87","#8db5bf","#004c6d","#6191a3","#badade","#eaffff","#F8FCFE"]
print ("###################Return Attribution###################")
df = metricsCalculator.getReturnAttribution(portfolioValue,rebalancing,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD)
df= (round((df/df.sum())*100,2))
df=pd.DataFrame(df)
print (df)
df.columns=["Return Attribution"]
df["Return Attribution"].plot.pie(autopct='%.2f',colors=colors, fontsize=15, figsize=(12, 12))
plt.title("Return Attribution",fontsize=25)
plt.show()


#Risk Attribution
colors=["#376e87","#8db5bf","#badade","#004c6d","#6191a3","#eaffff"]
print ()
print ("###################Risk Attribution###################")
riskAttribution = metricsCalculator.getRiskAttribution(portfolioValue, rtnBreakDown,rtnBreakDownCAD,exposure,'2020-06-01')
print (round(riskAttribution*100,2))
riskAttribution["Risk Attribution"].plot.pie(autopct='%.2f', colors=colors,fontsize=15, figsize=(12, 12))
plt.title("Risk Attribution",fontsize=25)
plt.show()



#####################################################
''' Risk Model'''
#Linear Model and then shocking one by one and also, copula based distribution
portReturns=benchmarkData.Port_Returns.dropna()
upScenario,downScenario,simup,simdown=riskModel.getResults(benchmarkData,portfolioValue,tickerAlts,tickerCredit,tickerEquity,tickerHedge,tickerAltsCAD,tickerCreditCAD,tickerEquityCAD,tickerHedgeCAD)
upScenario=upScenario.drop(["constant"],axis=1)
upScenario["Value"]=upScenario["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
upScenario["Portfolio Estimated Return"]=upScenario["Portfolio Estimated Return"]*100

utilityFuncs.goodPrint(round(upScenario,2))

downScenario=downScenario.drop(["constant"],axis=1)
downScenario["Value"]=downScenario["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
downScenario["Portfolio Estimated Return"]=downScenario["Portfolio Estimated Return"]*100
utilityFuncs.goodPrint(round(downScenario,2))

simup["Value"]=simup["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
simup["Portfolio Estimated Return"]=simup["Portfolio Estimated Return"]*100

simdown["Value"]=simdown["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
simdown["Portfolio Estimated Return"]=simdown["Portfolio Estimated Return"]*100

simup=simup.drop(["constant"],axis=1)
simdown=simdown.drop(["constant"],axis=1)

utilityFuncs.goodPrint(round(simup,2))
utilityFuncs.goodPrint(round(simdown,2))
##################################################

''' Stressed VaR '''

returns = [rtnBreakDown[0], rtnBreakDown[1], rtnBreakDown[2], rtnBreakDownCAD[0], rtnBreakDownCAD[1], rtnBreakDownCAD[2]]
name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
returns = pd.DataFrame(returns).T.dropna()
returns.columns = name

monthlyReturns = returns

FF5 = pd.read_csv('Data/F-F_Research_Data_5_Factors_2x3_daily.CSV')
FF5.columns = ['Date']+list(FF5.columns[1:])
FF5.Date = FF5.Date.apply(lambda x:str(x))
FF5.Date = FF5.Date.apply(lambda x:x[0:4]+'-'+x[4:6]+'-'+x[6:])
FF5.Date = pd.to_datetime(FF5.Date)
FF5.set_index('Date',inplace=True)

# FF5.insert(0,'constant',1)
FF5['RF'].shape
monthlyReturns.shape

df = monthlyReturns.join(FF5).dropna()
df[monthlyReturns.columns] = df[monthlyReturns.columns].sub(FF5['RF'],axis=0)


X=df[FF5.columns[:-1]]
X = sm.add_constant(X)
betaList = []


for i in range(monthlyReturns.shape[1]):
    Y=df.iloc[:,i]
    model = sm.OLS(Y, X).fit()
    betaList.append(model.params)
betaList = pd.DataFrame(betaList).T
betaList.columns = monthlyReturns.columns

HedgeTicker=["CGL.TO","IEF"]

Hedge = portfolioValue[HedgeTicker].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
Hedge = Hedge.replace([0,-1],np.nan).dropna()
df2 = Hedge.join(FF5)
df2[Hedge.columns] = df2[Hedge.columns].sub(FF5['RF'],axis=0)
X=df2[FF5.columns[:-1]]
X = sm.add_constant(X)
betaList2 = []


for i in range(Hedge.shape[1]):
    Y=df2.iloc[:,i]
    model = sm.OLS(Y, X).fit()
    betaList2.append(model.params)
betaList2 = pd.DataFrame(betaList2).T
betaList2.columns = Hedge.columns

betaList = betaList.join(betaList2)

def stressVaR(start_date,end_date,quantile):
    FF = sm.add_constant(FF5.loc[start_date:end_date])

    weight = (metricsCalculator.getExposure(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD,tickerHedge,tickerHedgeCAD,date=list(portfolioValue.loc[portfolioValue['CGL.TO']!=0].index[1:])[np.random.choice(portfolioValue.loc[portfolioValue['CGL.TO']!=0].index[1:].shape[0])]))
    np.array(weight.iloc[:-1])

    returns = np.dot(np.array(FF.iloc[:,:-1]),np.array(betaList))
    # get 99% VaR
    return np.sort(returns.dot(np.array(weight.iloc[:-1])),axis=0)[round(len(returns.dot(np.array(weight.iloc[:-1])))*(1-quantile))]

subprimeRecession_99 = stressVaR('2008-01','2009-06',0.99)
subprimeRecession_95 = stressVaR('2008-01','2009-06',0.95)
subprimeRecession_90 = stressVaR('2008-01','2009-06',0.90)

# 2001 tech bubble
techBubble_99 = stressVaR('2000-03','2002-09',0.99)
techBubble_95 = stressVaR('2000-03','2002-09',0.95)
techBubble_90 = stressVaR('2000-03','2002-09',0.90)
# 911
sellOff911_99 = stressVaR('2001-07','2001-09',0.99)
sellOff911_95 = stressVaR('2001-07','2001-09',0.95)
sellOff911_90 = stressVaR('2001-07','2001-09',0.90)

# Asian crisis
AsianCrisis_99 = stressVaR('1998-04','1998-10',0.99)
AsianCrisis_95 = stressVaR('1998-04','1998-10',0.95)
AsianCrisis_90 = stressVaR('1998-04','1998-10',0.90)

# Summer 2011
Summer2011_99 = stressVaR('2011-06','2011-10',0.99)
Summer2011_95 = stressVaR('2011-06','2011-10',0.95)
Summer2011_90 = stressVaR('2011-06','2011-10',0.90)

# 2015-2016 growth scare
growthScare_99 = stressVaR('2015-06','2016-1',0.99)
growthScare_95 = stressVaR('2015-06','2016-1',0.95)
growthScare_90 = stressVaR('2015-06','2016-1',0.90)

labels = ['Asian crisis', '2001 Tech Bubble', '9/11 sell out', 'Subprime crisis', 'Summer 2011', 'Growth scare 15-16']
VaR_90 = [AsianCrisis_90, techBubble_90, sellOff911_90, subprimeRecession_90, Summer2011_90, growthScare_90]
VaR_95 = [AsianCrisis_95, techBubble_95, sellOff911_95, subprimeRecession_95, Summer2011_95, growthScare_95]
VaR_99 = [AsianCrisis_99, techBubble_99, sellOff911_99, subprimeRecession_99, Summer2011_99, growthScare_99]
VaR_90 = [100*VaR_90[i][0] for i in range(len(VaR_90))]
VaR_95 = [100*VaR_95[i][0] for i in range(len(VaR_95))]
VaR_99 = [100*VaR_99[i][0] for i in range(len(VaR_99))]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, VaR_90, width, label='90% VaR')
rects2 = ax.bar(x + width/2, VaR_95, width, label='95% VaR')
rects2 = ax.bar(x + width*1.5, VaR_99, width, label='99% VaR')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('VaR Returns (%)')
ax.set_title('Stress Value at Risk')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()



#10 day VaR
returns = [rtnBreakDown[0], rtnBreakDown[1], rtnBreakDown[2], rtnBreakDownCAD[0], rtnBreakDownCAD[1], rtnBreakDownCAD[2]]
name = ['EQ_USD', 'CR_USD', 'Alt_USD', 'EQ_CAD', 'CR_CAD', 'Alt_CAD']
returns = pd.DataFrame(returns).T.dropna()
returns.columns = name

monthlyReturns = (returns+1).resample('10d').prod()-1
FF5 = pd.read_csv('Data/F-F_Research_Data_5_Factors_2x3_daily.CSV')
FF5.columns = ['Date']+list(FF5.columns[1:])
FF5.Date = FF5.Date.apply(lambda x:str(x))
FF5.Date = FF5.Date.apply(lambda x:x[0:4]+'-'+x[4:6]+'-'+x[6:])
FF5.Date = pd.to_datetime(FF5.Date)

FF5.set_index('Date',inplace=True)
FF5 = (FF5+1).resample('10d').prod()-1
FF = FF5.loc['2010-10-01':]
FF.index = monthlyReturns.index


df = monthlyReturns.join(FF).dropna()
df[monthlyReturns.columns] = df[monthlyReturns.columns].sub(FF['RF'],axis=0)

X=df[FF.columns[:-1]]
X = sm.add_constant(X)
betaList = []

for i in range(monthlyReturns.shape[1]):
    Y=df.iloc[:,i]
    model = sm.OLS(Y, X).fit()
    betaList.append(model.params)
    
betaList = pd.DataFrame(betaList).T
betaList.columns = monthlyReturns.columns

Hedge = portfolioValue[HedgeTicker].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
Hedge = Hedge.replace([0,-1],np.nan).dropna()
Hedge = (Hedge+1).resample('10d').prod()-1
Hedge.index = FF.loc['2015-07-20':'2020-02-15'].index
df2 = Hedge.join(FF.loc['2015-07-20':'2020-02-15'])
df2[Hedge.columns] = df2[Hedge.columns].sub(FF['RF'],axis=0)
X=df2[FF5.columns[:-1]]
X = sm.add_constant(X)
betaList2 = []

for i in range(Hedge.shape[1]):
    Y=df2.iloc[:,i]
    model = sm.OLS(Y, X).fit()
    betaList2.append(model.params)
    
betaList2 = pd.DataFrame(betaList2).T
betaList2.columns = Hedge.columns

betaList = betaList.join(betaList2)

subprimeRecession_99 = stressVaR('2008-01','2009-06',0.99)
subprimeRecession_95 = stressVaR('2008-01','2009-06',0.95)
subprimeRecession_90 = stressVaR('2008-01','2009-06',0.90)

# 2001 tech bubble
techBubble_99 = stressVaR('2000-03','2002-09',0.99)
techBubble_95 = stressVaR('2000-03','2002-09',0.95)
techBubble_90 = stressVaR('2000-03','2002-09',0.90)
# 911
sellOff911_99 = stressVaR('2001-07','2001-09',0.99)
sellOff911_95 = stressVaR('2001-07','2001-09',0.95)
sellOff911_90 = stressVaR('2001-07','2001-09',0.90)

# Asian crisis
AsianCrisis_99 = stressVaR('1998-04','1998-10',0.99)
AsianCrisis_95 = stressVaR('1998-04','1998-10',0.95)
AsianCrisis_90 = stressVaR('1998-04','1998-10',0.90)

# Summer 2011
Summer2011_99 = stressVaR('2011-06','2011-10',0.99)
Summer2011_95 = stressVaR('2011-06','2011-10',0.95)
Summer2011_90 = stressVaR('2011-06','2011-10',0.90)

# 2015-2016 growth scare
growthScare_99 = stressVaR('2015-06','2016-1',0.99)
growthScare_95 = stressVaR('2015-06','2016-1',0.95)
growthScare_90 = stressVaR('2015-06','2016-1',0.90)



labels = ['Asian crisis', '2001 tech bubble', '911 sell out', 'Subprime crisis', 'Summer2011', 'Growth scare']
VaR_90 = [AsianCrisis_90, techBubble_90, sellOff911_90, subprimeRecession_90, Summer2011_90, growthScare_90]
VaR_95 = [AsianCrisis_95, techBubble_95, sellOff911_95, subprimeRecession_95, Summer2011_95, growthScare_95]
VaR_99 = [AsianCrisis_99, techBubble_99, sellOff911_99, subprimeRecession_99, Summer2011_99, growthScare_99]
VaR_90 = [100*VaR_90[i][0] for i in range(len(VaR_90))]
VaR_95 = [100*VaR_95[i][0] for i in range(len(VaR_95))]
VaR_99 = [100*VaR_99[i][0] for i in range(len(VaR_99))]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars



fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, VaR_90, width, label='90% VaR')
rects2 = ax.bar(x + width/2, VaR_95, width, label='95% VaR')
rects2 = ax.bar(x + width*1.5, VaR_99, width, label='99% VaR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('VaR Returns (%)')
ax.set_title('Stress 10-day Value at Risk')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()



