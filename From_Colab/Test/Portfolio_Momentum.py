'''
전략 : 모멘텀 - 최근에 가장 많이 오른 종목 매수 후 일정 기간 보유한 후 팜 (60일 영업일 수익률)
알고리즘 : PyportfolioOpt 라이브러리 이용한 최적화
(max sharp, risk, return, fund remaining)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from pykrx import stock
import datetime
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


# 오늘 KOSPI&KOSDAQ 종목 전체 불러오기
today = datetime.datetime.today().strftime("%Y%m%d")
kospi = stock.get_market_fundamental_by_ticker(today, market='KOSPI').index
kosdaq = stock.get_market_fundamental_by_ticker(today, market='KOSDAQ').index
stocks = kospi.append(kosdaq)

def momentum_1month(stocks):  # 종목 list넣으면, 모멘텀 순위 있는 데이터프레임 출력
    df = pd.DataFrame()
    for s in stocks:
        df[s] = fdr.DataReader(s, '2021-01-01')['Close']

    # 20 영업일 수익률
    return_df = df.pct_change(20)
    return_df

    # 오늘 날짜
    today = datetime.datetime.today().strftime("%Y-%m-%d")

    # index는 종목 코드이고 모멘텀 데이터 있는 데이터 프레임으로 만들기
    s = return_df.loc[today]
    momentum_df = pd.DataFrame(s)
    momentum_df.columns = ["모멘텀"]

    momentum_df['순위'] = momentum_df['모멘텀'].rank(ascending=False)
    momentum_df = momentum_df.sort_values(by='순위')
    return momentum_df  # 모멘텀

def momentum_3months(stocks):  # 종목 list넣으면, 모멘텀 순위 있는 데이터프레임 출력
    df = pd.DataFrame()
    for s in stocks:
        df[s] = fdr.DataReader(s, '2021-01-01')['Close']

    # 60 영업일 수익률
    return_df = df.pct_change(60)
    return_df

    # 오늘 날짜
    today = datetime.datetime.today().strftime("%Y-%m-%d")

    # index는 종목 코드이고 모멘텀 데이터 있는 데이터 프레임으로 만들기
    s = return_df.loc[today]
    momentum_df = pd.DataFrame(s)
    momentum_df.columns = ["모멘텀"]

    momentum_df['순위'] = momentum_df['모멘텀'].rank(ascending=False)
    momentum_df = momentum_df.sort_values(by='순위')
    return momentum_df  # 모멘텀

momentum_1month_rank  = momentum_1month(stocks)
momentum_3months_rank = momentum_3months(stocks)

assets_1mo  = np.array(momentum_1month_rank.index[:30]) # top30개
assets_3mos = np.array(momentum_3months_rank.index[:30])
start_date  = '2018-07-19'
end_date    = '2021-07-19'
df_1mo      = pd.DataFrame()
df_3mos     = pd.DataFrame()
for s in assets_1mo:
  df_1mo[s] = fdr.DataReader(s, start_date, end_date)['Close']
df_1mo.info()
for stock in assets_3mos:
  df_3mos[stock] = fdr.DataReader(stock, start_date, end_date)['Close']
df_3mos.info()

# drop null
df_1mo_dropna  = df_1mo.dropna(axis=1)
df_3mos_dropna = df_3mos.dropna(axis=1)

# Annual return, covariance + optimize portfolio
mu_1mo  = expected_returns.mean_historical_return(df_1mo_dropna)
mu_3mos = expected_returns.mean_historical_return(df_3mos_dropna)
S_1mo   = risk_models.sample_cov(df_1mo_dropna)
S_3mos  = risk_models.sample_cov(df_3mos_dropna)
ef_1mo  = EfficientFrontier(mu_1mo, S_1mo, solver="SCS")
ef_3mos = EfficientFrontier(mu_3mos, S_3mos, solver="SCS")
weights_1mo          = ef_1mo.max_sharpe()
weights_3mos         = ef_3mos.max_sharpe()
cleaned_weights_1mo  = ef_1mo.clean_weights()
cleaned_weights_3mos = ef_3mos.clean_weights()
print(ef_1mo.portfolio_performance(verbose=True))
print(ef_3mos.portfolio_performance(verbose=True))

portfolio_val      = 15000000
latest_prices_1mo  = get_latest_prices(df_1mo_dropna)
latest_prices_3mos = get_latest_prices(df_3mos_dropna)
weights_1mo        = cleaned_weights_1mo
weights_3mos       = cleaned_weights_3mos
da_1mo            = DiscreteAllocation(weights_1mo, latest_prices_1mo, total_portfolio_value=portfolio_val)
da_3mos            = DiscreteAllocation(weights_3mos, latest_prices_3mos, total_portfolio_value=portfolio_val)
allocation_1mo, leftover_1mo    = da_1mo.lp_portfolio(verbose=False)
allocation_3mos, leftover_3mos  = da_3mos.lp_portfolio(verbose=False)
rmse_1mo           = da_1mo._allocation_rmse_error(verbose=False)
rmse_3mos          = da_3mos._allocation_rmse_error(verbose=False)

print('할당량 1개월: ', allocation_1mo)
print('할당량 3개월: ', allocation_3mos)
print('잔금 1개월: ', leftover_1mo, ' KRW')
print('잔금 3개월: ', leftover_3mos, ' KRW')

# Get the discrete allocation values
discrete_allocation_list_1mo = []
discrete_allocation_list_3mos = []
for symbol in allocation_1mo:
  discrete_allocation_list_1mo.append(allocation_1mo.get(symbol))
for symbol in allocation_3mos:
  discrete_allocation_list_3mos.append(allocation_3mos.get(symbol))

portfolio_1mo  = pd.DataFrame(columns = ['company_Ticker', 'Discrete_val_'+str(portfolio_val)])
portfolio_3mos = pd.DataFrame(columns = ['company_Ticker', 'Discrete_val_'+str(portfolio_val)])
portfolio_1mo['company_Ticker']  = allocation_1mo
portfolio_3mos['company_Ticker'] = allocation_3mos
portfolio_1mo['Discrete_val_'+str(portfolio_val)]  = discrete_allocation_list_1mo
portfolio_3mos['Discrete_val_'+str(portfolio_val)] = discrete_allocation_list_3mos
portfolio_1mo_sorted = portfolio_1mo.sort_values('Discrete_val_'+str(portfolio_val), ascending = False)
portfolio_1mo_sorted = portfolio_1mo_sorted.reset_index(drop=True)
portfolio_3mos_sorted = portfolio_3mos.sort_values('Discrete_val_'+str(portfolio_val), ascending = False)
portfolio_3mos_sorted = portfolio_3mos_sorted.reset_index(drop=True)

print('----- 1 month momentum portfolio performance -----')
print('잔금 1개월 : ', leftover_1mo, ' KRW')
print(ef_1mo.portfolio_performance(verbose=True))
print('Allocation has RMSE: {:.3f}'.format(rmse_1mo))

print('----- 3 months momentum portfolio performance -----')
print('잔금 3개월 : ', leftover_3mos, ' KRW')
print(ef_3mos.portfolio_performance(verbose=True))
print('Allocation has RMSE: {:.3f}'.format(rmse_3mos))

