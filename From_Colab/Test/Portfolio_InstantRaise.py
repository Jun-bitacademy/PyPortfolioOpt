'''
전략 : 급등주 포착 (직전날 대비 10배 이상 오른 종목 포착)
알고리즘 : PyportfolioOpt 라이브러리 이용한 최적화
(max sharp, risk, return, fund remaining)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from pykrx import stock
from datetime import datetime
import datetime # ???
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# 급등주 포착을 위한 함수
def check_speedy_rising_volume_yesterday(code): # 어제를 기준으로
    today   = datetime.datetime.today().strftime("%Y%m%d") # 20170303 같은 포맷
    df      = fdr.DataReader(code, '2020-01-01')
    volumes = df['Volume'].iloc[::-1]

    if len(volumes) < 22: # 총 22일 치의 데이터가 없을 경우 제외(최근 상장 종목)
        return False

    sum_vol20 = 0
    today_vol = 0

    for i, vol in enumerate(volumes): # 인덱스와 원소에 동시 접근
        if i == 0: # 오늘 날짜
            continue
        elif i == 1: # 어제 날짜
            today_vol = vol
        elif 2 <= i <= 21:
            sum_vol20 += vol
        else:
            break

    avg_vol20 = sum_vol20 / 20 # 최근 20일간 평균 거래량 구하기
    if today_vol > avg_vol20 * 10: # 조회 시작일의 거래량이 평균 거래량을 1000% 초과한다면 True
        return True

# 오늘 KOSPI&KOSDAQ 종목 전체 불러오기
today  = datetime.datetime.today().strftime("%Y%m%d")
kospi  = stock.get_market_fundamental_by_ticker(today, market='KOSPI').index
kosdaq = stock.get_market_fundamental_by_ticker(today, market='KOSDAQ').index
stocks = kospi.append(kosdaq)

def run(): # 어제 거래량이 1000% 오늘 종목 찾기
    speedy_rising_volume_list = []
    # num = len(stocks)

    for i, code in enumerate(stocks):
#         print(i, '/', num)
        if check_speedy_rising_volume_yesterday(code):
            print("급등주: ", code)
            speedy_rising_volume_list.append(code)
    return speedy_rising_volume_list

# 결과 = 급등주 : 종목코드 이런식으로 나옴. 생각보다 시간이 걸림 하나 나오는데.
speedy_rising_volume_list = run() # 급등주:  003720

# 급등주 종목 저장
assets = np.array(speedy_rising_volume_list) # 25개
print(assets)
print(len(assets))

# 3년치 주가 데이터 가져오기
start_date  = '2018-07-19'
end_date    = '2021-07-19'
df          = pd.DataFrame()
for stock in assets:
  df[stock] = fdr.DataReader(stock, start_date, end_date)['Close']
print(df)

# 결측값 포함 열 삭제
df_dropna = df.dropna(axis = 1)

# ------------- 포폴 최적화

# 복리 연평균 수익률
# return (1 + returns).prod() ** (frequency / returns.count()) - 1
mu = expected_returns.mean_historical_return(df_dropna)
S  = risk_models.sample_cov(df_dropna) # 21개개

ef = EfficientFrontier(mu, S, solver="SCS")
# 결과값 : CML(자본시장선)과 효율적 투자선(efficient frontier)의 접점에 있는 포트폴리오
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(ef.portfolio_performance(verbose=True))

# 종목별 할당량
portfolio_val = 15000000 # 투자금액 (단위: KRW)
latest_prices = get_latest_prices(df_dropna)
weights       = cleaned_weights
da            = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio(verbose=False)
print('Discrete Allocaion: ', allocation)
print('Funds Remaining: ', leftover, ' KRW')

'''
< 보류 > 
평균 제곱근 오차(Root Mean Square Error; RMSE)
포트폴리오의 종목별 비중과 실제 할당된 비중 간의 차이
rmse = da._allocation_rmse_error(verbose=False)
print('Allocation has RMSE: {:.5f}'.format(rmse))
'''

# 종목별 할당량 리스트화
discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))

portfolio = pd.DataFrame(columns = ['company_Ticker', 'Discrete_val_'+str(portfolio_val)])
portfolio['company_Ticker'] = allocation
portfolio['Discrete_val_'+str(portfolio_val)] = discrete_allocation_list
portfolio_sorted = portfolio.sort_values('Discrete_val_'+str(portfolio_val), ascending = False)
portfolio_sorted = portfolio_sorted.reset_index(drop=True)

# 잔금, 수익률 + 리스크 + 샤프지수
print('Funds Remaining: ', leftover, ' KRW')
print(ef.portfolio_performance(verbose=True))
