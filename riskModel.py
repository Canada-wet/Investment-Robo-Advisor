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
import metricsCalculator
import regimeDetection as rgd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copulas.multivariate import GaussianMultivariate
from copulas.multivariate import VineCopula

def getResults(benchmarkData,portfolioValue,tickerAlts,tickerCredit,tickerEquity,tickerHedge,tickerAltsCAD,tickerCreditCAD,tickerEquityCAD,tickerHedgeCAD):
    
    df = pd.read_csv("Data/MacroData.csv", index_col='DATE',parse_dates=True)
    df = df.loc[df.index>='2000-01-01'].iloc[:-2,:]
    df = df.iloc[:,:-1]
    df = df.applymap(lambda x:float(x)).dropna()
    credit_risk_premium = (df['BAMLC0A4CBBBEY']-df['BAMLC0A1CAAAEY'])
    inflation = df['CPIAUCSL'].pct_change().dropna()*100*12
    Industrial_prod_growth = df['INDPRO'].pct_change().dropna()*100
    riskData = pd.DataFrame(inflation).join(Industrial_prod_growth).join(df.iloc[:,2:7]).join(df.iloc[:,9:])
    riskData['CreditPremium'] = credit_risk_premium
    riskData.columns = ['Inflation','IndustrialProdGrowth','T-Bill','Oil','Libor','House','Unemploy','10 Yield curve','Term Premium','5 Yield Curve','2 Yield Curve','1 Yield Curve','CreditPremium']
    riskData['Unexpected Inflation'] = (riskData['Inflation']-riskData['Inflation'].shift(1))-(riskData['T-Bill'].shift(1)-riskData['T-Bill'].shift(2))
    riskData['Yield spread'] = riskData['10 Yield curve'] - riskData['T-Bill']
    riskData = riskData.dropna()
    riskData = riskData[['IndustrialProdGrowth','CreditPremium','10 Yield curve','T-Bill','Yield spread','5 Yield Curve','2 Yield Curve','1 Yield Curve']]
    x = riskData[['5 Yield Curve','2 Yield Curve','1 Yield Curve','T-Bill','10 Yield curve']]

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component'])
    riskData = riskData[['IndustrialProdGrowth','CreditPremium','Yield spread']]
    riskData['Yield Curve PCA']=principalDf.values
    riskfactorData=riskData.loc["5-2015":"7-2020"]
    
    riskfactorData = riskfactorData[['IndustrialProdGrowth','CreditPremium','Yield spread','Yield Curve PCA']]
    stocks = ["SPY"]
    start = datetime(2015,3,1)
    end = datetime(2020,5,31)
    factorData = pdr.get_data_yahoo(stocks, start=start, end=end)["Adj Close"]
    monthlySP=factorData.resample("m").last()
    monthlySP=monthlySP.pct_change().dropna()
    riskfactorData['SP500 Return'] = list(monthlySP.iloc[:-1].SPY)
    # riskfactorData['Yield Curve PCA'] = list(principalDf.values)
    riskfactorData["Port_Returns"]=list(benchmarkData.iloc[:-1].Port_Returns.dropna())

    X=riskfactorData[['IndustrialProdGrowth','CreditPremium','Yield Curve PCA','Yield spread','SP500 Return']]
    Y=riskfactorData['Port_Returns']
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    
    
    # Basic correlogram, THIS CAN GO IN OUR REPORT
    # sns.pairplot(X.join(Y))
    # plt.show()
    
    # Convert USD to CAD
    US_ticker = tickerAlts+tickerCredit+tickerEquity+tickerHedge
    portfolioValue[US_ticker] = portfolioValue[US_ticker].multiply(portfolioValue['Adj Close'],axis=0)
    
    # sub class
    EQ_ticker = tickerEquity+tickerEquityCAD
    CR_ticker = tickerCredit+tickerCreditCAD
    Alt_ticker = tickerAlts+tickerAltsCAD
    Hedge_ticker = tickerHedge+tickerHedgeCAD
    
    
    # normal scenario
    norm_scenario = riskfactorData.median()
    
    upScenario1 = pd.DataFrame(norm_scenario).T.copy()
    upScenario2 = pd.DataFrame(norm_scenario).T.copy()
    upScenario3 = pd.DataFrame(norm_scenario).T.copy()
    upScenario4 = pd.DataFrame(norm_scenario).T.copy()
    upScenario5 = pd.DataFrame(norm_scenario).T.copy()
    upScenarioBest = pd.DataFrame([2,2,1,-3,0.15,0]).T # best scenario
    upScenarioBest.columns = upScenario5.columns
    # shock one macroeconomic factor each time
    upScenario1['SP500 Return'] = 0.15
    upScenario2['IndustrialProdGrowth'] = 2
    upScenario3['CreditPremium'] = 2
    upScenario4['Yield spread'] = 1
    upScenario5['Yield Curve PCA'] = -3
    upScenario = pd.concat([upScenario1,upScenario2,upScenario3,upScenario4,upScenario5,upScenarioBest],axis=0)
    upScenario = upScenario.iloc[:,0:5]

    downScenario1 = pd.DataFrame(norm_scenario).T.copy()
    downScenario2 = pd.DataFrame(norm_scenario).T.copy()
    downScenario3 = pd.DataFrame(norm_scenario).T.copy()
    downScenario4 = pd.DataFrame(norm_scenario).T.copy()
    downScenario5 = pd.DataFrame(norm_scenario).T.copy()
    # shock one macroeconomic factor each time
    downScenario1['SP500 Return'] = -0.25
    downScenario2['IndustrialProdGrowth'] = -10
    downScenario3['CreditPremium'] = -2
    downScenario4['Yield spread'] = -2
    downScenario5['Yield Curve PCA'] = 1
    downScenarioWorst = pd.DataFrame([-10,-2,-2,1,-0.25,0]).T # worst scenario
    downScenarioWorst.columns = downScenario5.columns
    downScenario = pd.concat([downScenario1,downScenario2,downScenario3,downScenario4,downScenario5,downScenarioWorst],axis=0)
    downScenario = downScenario.iloc[:,0:5]
    
    z = riskfactorData.shape[1] # number of independent variables
    
    downScenario.insert(0,'constant',1)
    upScenario.insert(0,'constant',1)
    upPortfolio = np.dot(np.array(upScenario),np.array(model.params))
    upScenario['Portfolio Estimated Return'] = upPortfolio
    downPortfolio = np.dot(np.array(downScenario),np.array(model.params))
    downScenario['Portfolio Estimated Return'] = downPortfolio
    # downScenario = downScenario.loc[downScenario['Portfolio Estimated Return']<=downThreshold]
    

    data=riskfactorData[riskfactorData.columns[:-1]]
    # copula = GaussianMultivariate()
    # copula = VineCopula('center')
    copula = VineCopula('regular')
    # copula = VineCopula('direct')
    copula.fit(data)
    samples = copula.sample(10000)
    samples.insert(0,'constant',1)
    samplePortfolio = np.dot(np.array(samples),np.array(model.params))
    samples['Portfolio Estimated Return'] = samplePortfolio
    upScenario1 = samples.sort_values('Portfolio Estimated Return',ascending=0).iloc[:3,:]
    downScenario1 = samples.sort_values('Portfolio Estimated Return',ascending=1).iloc[:3,:]
   
    return upScenario,downScenario,upScenario1,downScenario1
    
