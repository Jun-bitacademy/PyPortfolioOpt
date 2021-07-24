import pandas as pd
import numpy as np
import requests

# finance-datareader installed
# now : cvxopt version 1.2.6
# python interpreter : 3.9.6

import FinanceDataReader as fdr
# print(fdr.__version__) # 0.9.31

# 한국거래소 krx 불러오기
df_krx = fdr.StockListing('KRX')
# print(df_krx) # [6813 rows x 10 columns]

# 데이터 파악
# df_krx.info()
# df_krx.isnull().sum() # 이거 안됨.

# 결측치 제거
df_krx_dropna = df_krx.dropna()
# df_krx_dropna.info() #  0   Symbol          2256 non-null   object

# 종목코드 가져오기
assets = df_krx_dropna['Symbol']
# print(assets) # 가능

# 가져온 종목 코드 array 안에 담기
assets = np.array(assets)
# print(len(assets)) # 2256
# ---------------------------------------
# 종목별 종가 가져오기
from datetime import datetime

# 주식 시작일은 2013년 1월 1일이고 종료일은 현재 날짜 (오늘)로 설정
#Get the stock starting date
start_date = '2013-01-01'
# today = datetime.today().strftime('%Y-%m-%d')
end_date = '2021-07-16' # datetime.today().strftime('%Y-%m-%d')

# 각 주식의 일별 종가 데이터를 저장할 데이터 프레임을 생성
#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

# FinanceDataReader로 각 종목의 종가데이터 불러오기
for stock in assets:
  df[stock] = fdr.DataReader(stock, start_date, end_date)['Close']
# print(df) # [2102 rows x 2256 columns] 시간 오지게 오래걸림. 최소 5분은 걸린듯. 밑에 같은 warning 을 줌. 버그는 아님. 워닝을 보기 싫으면 pandas를 downgrade
# PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.
# Consider using pd.concat instead.  To get a de-fragmented frame, use `newframe = frame.copy()`
#   df[stock] = fdr.DataReader(stock, start_date, end_date)['Close']


# DataFrame을 csv 파일로 저장하기 ( 결측값 제거하지 않음 )
# df.to_csv("krx_code_close.csv", index=True)

# 칼럼명을 회사이름으로 변경
df.columns = df_krx_dropna['Name'].values

# 결측값 있는 열 삭제  ( 종목 2256 -> 1476으로 줄어 듦 )
df2 = df.dropna(axis = 1)
# print(df2)

# 결측값을 가진 열을 제거한 DataFrame을 csv 파일로 저장하기
# df2.to_csv("krx_name_close_drop_columns.csv", index=True)

# Get the assets / tickers
assets = df2.columns
print(len(assets)) # 1476

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Calculate the expected annualized returns and the annualized sample covariance matrix of the daily asset returns
mu = expected_returns.mean_historical_return(df2)
S = risk_models.sample_cov(df2)

# Optimize for the maximal Sharpe ratio
# 💛데이터셋이 너무 많으면, ef.max_sharpe()에서 에러남 -> solver를 SCS로 바꿔줌
# Rober says: 100개 이하로 종목을 추린 후에 실행시키기를 추천함 !
ef = EfficientFrontier(mu, S, solver="SCS")  # Create the Efficient Frontier Object

# Maximize the Sharpe ratio, and get the raw weights
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

ef.portfolio_performance(verbose=True)

# Get the discrete allocation of each sharpe per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# 투자금액 (단위: KRW)
portfolio_val = 5000000

latest_prices = get_latest_prices(df2)

weights = cleaned_weights

da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)

allocation, leftover = da.lp_portfolio()

print('Discrete Allocaion: ', allocation) # 종목당 몇주 살지 추천 결과 : 38개
print('Funds Remaining: ', leftover, ' KRW')

# 포트폴리오에 포함된 종목을 리스트로 만들기 (회사 이름만 리스트에 담김)
company_name = list(allocation)

# Get the discrete allocation values (리스트안에 담긴 숫자들만 나열)
discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))

# Create a dataframe for the portfolio
portfolio_df = pd.DataFrame(columns=['Company_name', 'company_Ticker', 'Discrete_val_' + str(portfolio_val)])
# 결과: Company_name	company_Ticker	Discrete_val_5000000

portfolio_df['Company_name'] = company_name
portfolio_df['company_Ticker'] = allocation # 원래 종목 코드여야 하는데 앞에서 컬럼 수정을 해버려서 그런것임.
portfolio_df['Discrete_val_'+str(portfolio_val)] = discrete_allocation_list
# print(portfolio_df)

# Show Funds Remaining
print('Funds Remaining: ', leftover, ' KRW')

# Show Portfolio performance
print(ef.portfolio_performance(verbose=True))

# 총 3-5분정도 걸린듯
# 산업별 코드를 돌려봐야 한다. 그 코드는 만들어야 할듯.. ?