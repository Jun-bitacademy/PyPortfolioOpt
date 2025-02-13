'''
‘급등주 포착’ 알고리즘은 특정 거래일의 거래량이 이전 시점의 평균 거래량보다 1,000% 이상 급증하는 종목을 매수
‘이전 시점의 평균 거래량’을 특정 거래일 이전의 20일(거래일 기준) 동안의 평균 거래량으로 정의
‘거래량 급증’은 특정 거래일의 거래량이 평균 거래량보다 1,000% 초과일 때 급등한 것으로 정의
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from pykrx import stock
import datetime
import requests
# from datetime import timedelta # 마이크로초 전, 마이크로초 후 를 구하고 싶다면 timedelta
from dateutil.relativedelta import relativedelta # 몇달 전, 몇달 후, 몇년 전, 몇년 후 를 구하고 싶다면 relativedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
import warnings
warnings.filterwarnings(action='ignore')

def check_speedy_rising_volume_yesterday(code): # 어제를 기준으로
    today = datetime.datetime.today().strftime("%Y%m%d")
    df = fdr.DataReader(code, '2020-01-01')
    volumes = df['Volume'].iloc[::-1]

    if len(volumes) < 22: # 총 22일 치의 데이터가 없을 경우 제외(최근 상장 종목)
        return False

    sum_vol20 = 0
    today_vol = 0

    for i, vol in enumerate(volumes):
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
today = datetime.datetime.today().strftime("%Y%m%d")
kospi = stock.get_market_fundamental_by_ticker(today, market='KOSPI').index
kosdaq = stock.get_market_fundamental_by_ticker(today, market='KOSDAQ').index
stocks = kospi.append(kosdaq)

def run(): # 어제 거래량이 1000% 오늘 종목 찾기
    speedy_rising_volume_list = []
    num = len(stocks)

    for i, code in enumerate(stocks):
        if check_speedy_rising_volume_yesterday(code):
            #print("급등주: ", code)
            speedy_rising_volume_list.append(code)
    return speedy_rising_volume_list

speedy_rising_volume_list = run()

speedy_rising_volume_list_df = pd.DataFrame({'speedy_rising_volume_list':speedy_rising_volume_list})
# 종목코드 6자리로 맞춰주기
speedy_rising_volume_list_df['speedy_rising_volume_list'] = speedy_rising_volume_list_df['speedy_rising_volume_list'].apply(lambda x: '{0:0>6}'.format(x))

# 종목 이름 및 코드
kospi_temp = fdr.StockListing('KOSPI')[['Symbol', 'Name']]
kosdaq_temp = fdr.StockListing('KOSDAQ')[['Symbol', 'Name']]
code_name_dict = pd.concat([kospi_temp,kosdaq_temp])
code_name_dict = code_name_dict.set_index('Symbol').to_dict().get('Name') # {'095570': 'AJ네트웍스',

assets= np.array(speedy_rising_volume_list_df['speedy_rising_volume_list'].values)
start_date  = datetime.datetime.today() - relativedelta(years=3)
start_date  = start_date.strftime('%Y%m%d')
end_date    = today
df          = pd.DataFrame()

for s in assets:
  df[s] = fdr.DataReader(s, start_date, end_date)['Close']
df.info()

# drop null
df_dropna  = df.dropna(axis=1)
# 수익률의 공분산
mu  = expected_returns.mean_historical_return(df_dropna)
S   = risk_models.sample_cov(df_dropna)
#print(plotting.plot_covariance(S))

# 포폴 최적화 (Max sharp ratio)
ef  = EfficientFrontier(mu, S, solver="SCS")
weights          = ef.max_sharpe()
cleaned_weights  = ef.clean_weights()
print(ef.portfolio_performance(verbose=True))

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
# plt.legend(loc='best')
# plt.show()
# # Find the tangency portfolio
# ef.max_sharpe()
# ret_tangent, std_tangent, _ = ef.portfolio_performance()  # 기대수익률 , Risk(변동성)
# ax.scatter(std_tangent, ret_tangent, marker="*", s=200, c="r", label="Max Sharpe")
# # Generate random portfolios
# n_samples = 20000
# w = np.random.dirichlet(np.ones(len(mu)), n_samples) # 난수로 20000세트의 투자비중 만들기
# rets = w.dot(mu)                                     # 기대수익률
# stds = np.sqrt(np.diag(w @ S @ w.T))                 # Risk(변동성)
# sharpes = rets / stds                                # 샤프비율
#
# ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
#
# # Output
# ax.set_title("Efficient Frontier with random portfolios")
# ax.legend()
# plt.tight_layout()
# plt.savefig("ef_scatter.png", dpi=200)
# plt.show()
# print(ef.portfolio_performance(verbose=True)) # max sharp ratio 최적화된 포폴 수익률 변동성 및 샾비율
# print(plotting.plot_weights(weights, ax = None) # 종목별 투자비중

# 각 주 할당량 구하러 가자
portfolio_val  = 15000000
latest_prices  = get_latest_prices(df_dropna)
weights        = cleaned_weights
da             = DiscreteAllocation(weights, latest_prices, total_portfolio_value=portfolio_val)
allocation, leftover = da.lp_portfolio(verbose=False)
rmse           = da._allocation_rmse_error(verbose=False)

print('할당량 1개월: ', allocation)
print('잔금 1개월: ', leftover, ' KRW')

# 각 종목별 실제 투자 금액
inv_total_price = {}
for i in allocation.keys():
  inv_total_price[i] = latest_prices.loc[i]*allocation[i]
inv_total_price

# 총 투자금액
investment = 0
for i in inv_total_price.values():
    investment += i
print(investment)

# 각 종목별 실제 투자 비중
inv_total_weight = {}
for i in allocation.keys():
  inv_total_weight[i] = inv_total_price[i]/investment
inv_total_weight

# 투자비중의 합계
investment_w = 0
for i in inv_total_weight.values():
    investment_w += i
print(investment_w)

# 결과값으로 불러올 값을 리스트로 저장
name_list = []                    # 종목명(회사이름)
total_price_stock =[]             # 각 종목별 실제 투자 금액
total_weight_stock = []           # 각 종목별 실제 투자 비중
for i in allocation.keys(): # i = 포트폴리오에 할당된 종목의 종목코드
    name_list.append(code_name_dict.get(i))
    total_price_stock.append(inv_total_price.get(i))
    total_weight_stock.append(inv_total_weight.get(i))

# Get the discrete allocation values
discrete_allocation_list = []
for symbol in allocation:
  discrete_allocation_list.append(allocation.get(symbol))
print(discrete_allocation_list)

portfolio_df = pd.DataFrame(columns = ['종목명','종목코드','수량(주)', '투자금액(원)','투자비중'])
portfolio_df['종목명'] = name_list
portfolio_df['종목코드'] = allocation
portfolio_df['수량(주)'] = discrete_allocation_list
portfolio_df['투자금액(원)'] = total_price_stock
portfolio_df['투자비중'] = total_weight_stock
portfolio_df_sorted = portfolio_df.sort_values('투자비중', ascending = False)
portfolio_df_sorted = portfolio_df_sorted.reset_index(drop=True)
# 투자 금액에 따라 최적화된 포트폴리오 종목별 수량
portfolio_df_sorted.loc["합계",2:] = portfolio_df_sorted.sum()

print('----- 1 month momentum portfolio performance -----')
print('잔금 1개월 : ', leftover, ' KRW')
print(ef.portfolio_performance(verbose=True))
# print('Allocation has RMSE: {:.3f}'.format(rmse))

################# 코스피랑 비교 ####################
# 각 일자별, 종목별 종가에 해당 weights를 곱해주기
for i, weight in cleaned_weights.items():
    df_dropna[i] = df_dropna[i]*weight

# 일자별 종목의 (종가*비중) 합계를 Port열에 저장
df_dropna['Port'] = df_dropna.sum(axis = 1)

# 일자별 종가의 전일대비 변동률(수익률)을 portfolio라는 데이터프레임으로 저장
portfolio = df_dropna[['Port']].pct_change()

# 코스피지수 불러오기
kospi = fdr.DataReader('KS11', start_date, end_date)[['Close']]

# 코스피지수의 변동률(수익률) 구하기
# 변동률(수익률) = (당일가격-전일가격) / 전일가격
# 7/20의 변동률(수익률) = (7/20 가격-7-19 가격) / 7/19 가격
kospi_pct = kospi.pct_change()

# 코스피와 포트폴리오 합치기
result = kospi_pct.join(portfolio)

# 1열을 0으로 (Nan 값을 0으로)
result.iloc[0] = 0

# 열 이름 변경
result.columns = ['KOSPI', 'PORTFOLIO']

# 1에서 시작해서, 전일대비 변동률(수익률)을 적용하여 수치화하기
wealth = (1+result).cumprod()

# 포트폴리오와 KOSPI 지수의 '누적 수익률 추이'를 시각화하여 비교
# matplotlib.pyplot 스타일시트 설정
# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(18,5))
# plt.plot(wealth.index, wealth.KOSPI , 'r', label='KOSPI')
# plt.plot(wealth.index, wealth.PORTFOLIO ,'b', label="PORTFOLIO(momentum_1month)")
# plt.grid(True)
# plt.title('Return Trend')
# plt.xlabel('Date',fontsize=18, labelpad=7)
# plt.ylabel('Return',fontsize=18, labelpad=7)
# plt.legend(loc='best')
# plt.show()

# 변동률 비교
# plt.figure(figsize=(18,10))
#
# plt.subplot(2,1,1)
# plt.title('Volatility Trend')
# plt.plot(result.index, result.KOSPI , 'r', label='KOSPI')
# plt.yticks([-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
# plt.grid(True)
# plt.ylabel('Volatility',fontsize=18, labelpad=7)
# plt.legend(loc='best')
#
# plt.subplot(2,1,2)
# plt.plot(result.index, result.PORTFOLIO ,'b', label="PORTFOLIO(momentum_1month)")
# plt.yticks([-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
# plt.ylabel('Volatility',fontsize=18, labelpad=7)
# plt.legend(loc='best')
#
# plt.grid(True)
# plt.show()

print('----- 1 months momentum portfolio performance -----')
print('잔금: ', leftover, ' KRW')
print(ef.portfolio_performance(verbose=True))