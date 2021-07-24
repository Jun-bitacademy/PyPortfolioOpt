'''
전략 : 하루의 주가를 놓고 보면 오른 경우가 42%, 내린 경우가 46%, 나머지 12%는 변동이 없다. 증명
알고리즘 : PyportfolioOpt 라이브러리 이용한 최적화
(max sharp, risk, return, fund remaining)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import datetime
from pykrx import stock
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


def up_down_zero(code):  # 종목과 연도에 맞는 상승/하락/변동 없는 날 수를 리스트 반환
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    year = today[0:4]
    month_day = today[4:]
    one_year_ago = str(int(year) - 1) + month_day

    data = fdr.DataReader(code, one_year_ago)[['Close']]
    data_rtn = data.pct_change()

    up = 0
    nothing = 0
    down = 0
    for i, date in enumerate(data.index):
        if data_rtn.Close.iloc[i] > 0:
            up = up + 1
        elif data_rtn.Close.iloc[i] == 0:
            nothing = nothing + 1
        else:
            down = down + 1

    total_days = len(data_rtn.index)
    return up / total_days, down / total_days, nothing / total_days


def get_up_down_zero_df(stocks):  # stocks 리스트를 넣으면, 상승/하락/변동없는 확률 데이터프레임 반환
    up_list = []
    down_list = []
    zero_list = []
    for i in stocks:
        temp = up_down_zero(i)
        up_list.append(temp[0])
        down_list.append(temp[1])
        zero_list.append(temp[2])

    # 데이터 프레임 만들기
    up_down_zero_df = pd.DataFrame()
    up_down_zero_df['종목 코드'] = stocks  # 종목코드
    up_down_zero_df['상승 확률'] = up_list  # 일간 변동률이 양수인 날의 수
    up_down_zero_df['하락 확률'] = down_list  # 일간 변동률이 음수인 날의 수
    up_down_zero_df['변동 없는 확률'] = zero_list  # 일간 변동률이 0인 날의 수

    up_down_zero_df['상승 확률 높은 순위'] = up_down_zero_df['상승 확률'].rank(ascending=False)
    up_down_zero_df = up_down_zero_df.sort_values(by='상승 확률 높은 순위')
    return up_down_zero_df

up_down_zero_df = get_up_down_zero_df(stocks)

symbol_udz = []
for i in idx_list:
    symbol_udz.append(up_down_zero_df.loc[i][0])
symbol_udz

# 급등주 종목 저장
assets = np.array(symbol_udz)
start_date = '2018-07-21'
end_date = '2021-07-21'
df = pd.DataFrame()
for stock in assets:
  df[stock] = fdr.DataReader(stock, start_date, end_date)['Close']

df_dropna = df.dropna(axis = 1)

mu = expected_returns.mean_historical_return(df_dropna)
S = risk_models.sample_cov(df_dropna)

ef = EfficientFrontier(mu, S, solver="SCS")
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(ef.portfolio_performance(verbose=True))

portfolio_val = 15000000
latest_prices = get_latest_prices(df_dropna)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio(verbose=False)
#rmse = da._allocation_rmse_error(verbose=False)

print('Discrete Allocaion: ', allocation)
print('Funds Remaining: ', leftover, ' KRW')

discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))

portfolio_df = pd.DataFrame(columns = ['company_Ticker', 'Discrete_val_'+str(portfolio_val)])
portfolio_df['company_Ticker'] = allocation
portfolio_df['Discrete_val_'+str(portfolio_val)] = discrete_allocation_list
portfolio_df_sorted = portfolio_df.sort_values('Discrete_val_'+str(portfolio_val), ascending = False)
portfolio_df_sorted = portfolio_df_sorted.reset_index(drop=True)

print('Funds Remaining: ', leftover, ' KRW')
print(ef.portfolio_performance(verbose=True))
print('Allocation has RMSE: {:.3f}'.format(rmse))