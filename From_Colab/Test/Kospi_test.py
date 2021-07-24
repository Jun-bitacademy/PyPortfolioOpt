# Kospi 상장종목에서 KRX 관리종목 (상장폐지 우려로 인한 거래 중지된) 제거

import pandas as pd
import numpy as np
import requests
import FinanceDataReader as fdr
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# KRX 상장폐지 종목 리스트 (delisting symbol)
krx_adm         = fdr.StockListing('KRX-ADMINISTRATIVE')
# print(krx_adm) # 116 rows × 4 columns
krx_delisting   = krx_adm['Symbol'] # in array

# 코스피 리스트 전체 7.21 기준
kospi_list      = fdr.StockListing('KOSPI') # (df_kospi)
# print(kospi_list) # [5414 rows x 10 columns]
kospi_tot       = kospi_list['Symbol']

# KOSPI 목록에 포함된 KRX 관리종목 리스트로 저장 (종목코드로)
dangers = []
for krx in krx_delisting.values:
  for kospi in kospi_tot.values:
    if krx == kospi:
      dangers.append(krx)

# KOSPI 목록에 포함된 KRX 관리종목의 인덱스 찾기 (몇번째 줄에 있는지)
idx = []
for danger in dangers:
  for kospi in kospi_tot.values:
    if danger == kospi:
      idx.append(kospi_list[kospi_tot == danger].index)
      # len(idx) : 14, result : [Int64Index([3498], dtype='int64'),...

# 인덱스 번호 --> 리스트화
idx_list = []
for i in range (0, len(idx)):
    idx_list.append(idx[i][0])

# 코스피 목록에 있는 krx 관리종목 드랍
kospi_new = kospi_list.drop(idx_list) # 4787 rows × 10 columns, df_kospi_new

# 결측치 제거를 위한 확인
kospi_new.info()
# nul = kospi_new.isnull().sum() # 그냥은 안나오고 꼭 매개변수로 받아서 프린트해야 함
# print(nul)

# 결측치 제거
kospi_new_dropna = kospi_new.dropna()
kospi_new_dropna.info() # 760 으로 동일

# --------------------- 위에까지는 빠르고 밑에는 시간이 다소 걸림 / 밑 : 종목별 코드 수집, 위 : 관리종목 전처리

# 종목코드 받아온후 종목별로 종가 가져오기 (데이터 프레임 형성)
assets      = np.array(kospi_new_dropna['Symbol']) # 760개 종목코드
start_date  = '2013-01-01' # 임의로 설정
end_date    = '2021-07-20'
df          = pd.DataFrame() # 빈프레임 생성
for stock in assets:
  df[stock] = fdr.DataReader(stock, start_date, end_date)['Close'] # 칼럼 추가

# 종목코드를 회사이름으로 변경 (여기서 ticker 나중에 가져올때, 종목코드가 이름으로 바뀌어버림) ★★★★★★ 해결요망
df.columns = kospi_new_dropna['Name'].values
# print(df) # [1454 rows x 760 columns]

# 결측값 포함된 열 삭제
df_drop = df.dropna(axis = 1)
# df_drop.info() # Columns: 672 entries, AJ네트웍스 to 흥국화재

# ---------------------- 포트폴리오 최적화

# 일일 종목들의 연간 수익률 및 공분산 (상관관계)
mu  = expected_returns.mean_historical_return(df_drop) # 연간 평균 수익률
S   = risk_models.sample_cov(df_drop) # 공분산

# 샤프지수 극대화된 포폴의 최적화
ef              = EfficientFrontier(mu, S, solver="SCS")
weights         = ef.max_sharpe() # 100개 이하의 종목 넣어야 에러가 안남.
cleaned_weights = ef.clean_weights() # weight 전부 0이 됨
# print(ef.portfolio_performance(verbose=True)) # 반환값 보여주기 위함
'''
< 위 print 결과 값 >
Expected annual return: 34.9%
Annual volatility: 13.7%
Sharpe Ratio: 2.41
(0.3490823360914377, 0.13680291496318722, 2.4055213748916944)
'''

# --------------------- 종목별 할당량

portfolio_val  = 15000000 # 투자금액 (단위: KRW)
latest_prices  = get_latest_prices(df_drop)
weights        = cleaned_weights
da             = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio() # make it integer

# print('종목별 할당량: ', allocation)
# print('잔금: ', leftover, ' KRW') # 118.0  KRW
# print(len(allocation)) # 24 개

company_name = list(allocation)

# 할당량 리스트화
discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))

# Create a dataframe for the portfolio
portfolio = pd.DataFrame(columns = ['Company_name', 'Ticker', 'Dis_val_' + str(portfolio_val) + 'KRW'])
portfolio['Company_name'] = company_name
portfolio['Ticker'] = allocation
portfolio['Dis_val_'+str(portfolio_val)] = discrete_allocation_list

print(portfolio)

# Sort by allocation & Show the portfolio
portfolio_sorted = portfolio.sort_values('Dis_val_'+str(portfolio_val), ascending = False)
portfolio_sorted = portfolio_sorted.reset_index(drop=True)

print(portfolio_sorted)
print('잔금: ', leftover, ' KRW')
print(ef.portfolio_performance(verbose=True))