# 지수크롤링

import pandas as pd
from pandas import DataFrame, Series
import requests as re
from bs4 import BeautifulSoup
import datetime as date
import time

folder_adress = 'C:/it'


def market_index_crawling():
    Data = DataFrame()

    url_dict = {'미국 USD': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_USDKRW',
                '일본 JPY': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_JPYKRW',
                '유럽연합 EUR': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_EURKRW',
                '중국 CNY': 'http://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_CNYKRW',
                'WTI': 'http://finance.naver.com/marketindex/worldDailyQuote.nhn?marketindexCd=OIL_CL&fdtc=2',
                '국제 금': 'http://finance.naver.com/marketindex/worldDailyQuote.nhn?marketindexCd=CMDT_GC&fdtc=2'}

    for key in url_dict.keys():

        date = []
        value = []

        for i in range(1, 1000):
            url = re.get(url_dict[key] + '&page=%s' % i)
            url = url.content

            html = BeautifulSoup(url, 'html.parser')

            tbody = html.find('tbody')
            tr = tbody.find_all('tr')


            if len(tbody.text.strip()) > 3:

                for r in tr:
                    temp_date = r.find('td', {'class': 'date'}).text.replace('.', '-').strip()
                    temp_value = r.find('td', {'class': 'num'}).text.strip()

                    date.append(temp_date)
                    value.append(temp_value)
            else:

                temp = DataFrame(value, index=date, columns=[key])

                Data = pd.merge(Data, temp, how='outer', left_index=True, right_index=True)

                print(key + '완료')
                time.sleep(10)
                break

    Data.to_csv('%s/market_index.csv' % (folder_adress))
    return Data


K = market_index_crawling()

# 금리가져오기

import pandas as pd
from pandas import DataFrame, Series
import requests as re
from bs4 import BeautifulSoup

url = 'http://finance.naver.com/marketindex/interestDailyQuote.nhn?marketindexCd=IRR_CD91&page=1'

crawling_list = ['IRR_CALL', ]


folder_adress = 'C:/it'


def interest_rates():
    data_dict = {'IRR_CD91': [],
                 'IRR_CALL': [],
                 'IRR_GOVT03Y': [],
                 'IRR_CORP03Y': []}

    label_list = ['IRR_CD91', 'IRR_CALL', 'IRR_GOVT03Y', 'IRR_CORP03Y']

    Data = DataFrame()

    for label in label_list:

        date_list = []

        try:
            for i in range(1, 700):
                url = re.get(
                    'http://finance.naver.com/marketindex/interestDailyQuote.nhn?marketindexCd=%s&page=%s' % (label, i))
                url = url.content

                soup = BeautifulSoup(url, 'html.parser')


                dates = soup.select('tr > td.date')


                try:
                    test = soup.find('tbody').find('tr').find('td', {'class': 'num'}).text
                except:
                    break


                for date in dates:
                    date_list.append(date.text.replace('.', '-').strip())

                rates = soup.find('tbody').find_all('tr')

                for rate in rates:
                    data_dict[label].append(rate.find('td', {'class': 'num'}).text.strip())

        except:
            print('Error')

        temp_dataframe = DataFrame(data_dict[label], index=date_list)
        Data = pd.merge(Data, temp_dataframe, how='outer', left_index=True, right_index=True)

        print(label)

    Data.columns = ['CD91', '콜 금리', '국고채 3년', '회사채 3년']
    Data.to_csv('%s/interest_rate.csv' % folder_adress)
    return Data


DATA = interest_rates()

# 크롤링 데이터 합치기
import pandas as pd
import FinanceDataReader as fdr
df1 = fdr.DataReader('KS11','20110401')['Close'] # 코스피 불러오기
# df1.to_excel('C:/it/kospi_01.xlsx')
df2 = fdr.DataReader('US500','20110401')['Close']
df3 = fdr.DataReader('IXIC','20110401')['Close']
df4 = fdr.DataReader('DJI','20110401')['Close'];df4
df = pd.read_excel("C:/it/market_index.xlsx")
df5 = pd.read_excel("C:/it/interest_rate1.xlsx")
df6 = pd.read_excel("C:/it/회사채3년.xlsx")
df7 = pd.read_excel("C:/it/콜금리.xlsx")
df8 = fdr.DataReader('KQ11','20110401')['Close']
df_list = [fdr.DataReader('KS11','20110401')['Close'],df];df_list
result_1 = pd.merge(df1,df,on='Date',how='left')
result_2 = pd.merge(result_1,df2,on='Date',how='left')
result_3 = pd.merge(result_2,df3,on='Date',how='left')
result_4 = pd.merge(result_3,df4,on='Date',how='left');result_4
result_5 = pd.merge(result_4,df5,on='Date',how='left');result_5
result_6 = pd.merge(result_5,df6,on='Date',how='left');result_6
result_7 = pd.merge(result_6,df7,on='Date',how='left');result_7
result_8 = pd.merge(df8,result_7,on='Date',how='left');result_8
result_8.columns = ['Date','KOSDAQ','KOSPI','USDKRW','JPYKRW','EURKRW','CNYKRW','WTI','GOLD','SP500','NASIXIC','DOW','국고채3년금리','회사채3년금리','콜금리'];result_8
df_last.corr()
result_8.to_excel('C:/it/result_8.xlsx')
df_last = pd.read_excel("C:/it/result_8.xlsx");df_last
df_last = df_last.fillna(method='ffill')

# 데이터 시각화
import pandas as pd
df = pd.read_excel("C:/it/result_8.xlsx")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

k = (df['KOSPI']/df['KOSPI'].loc[0]) *100
kd = (df['KOSDAQ']/df['KOSDAQ'].loc[0])*100
do = (df['DOW']/df['DOW'].loc[0])*100
uk = (df['USDKRW']/df['USDKRW'].loc[0])*100
go = (df['GOLD']/df['GOLD'].loc[0])*100

plt.figure(figsize=(9,5))
plt.plot(df['Date'],(df['KOSPI']/df['KOSPI'].iloc[0]) *100,'b--')
plt.plot(df['Date'],(df['DOW']/df['DOW'].iloc[0])*100,'r')
plt.plot(df['Date'],(df['USDKRW']/df['USDKRW'].loc[0])*100,'g-')
plt.plot(df['Date'],(df['GOLD']/df['GOLD'].iloc[0])*100,'y')
plt.ylabel('Rate')
plt.grid(True)
plt.legend(['KOSPI','DOW','USDKRW','GOLD'])
plt.show()

df2 = pd.DataFrame({'X':df['DOW'],'Y':df['KOSPI']})
regr = stats.linregress(df2.X,df2.Y);reg
df2 = df2.fillna(method='bfill')
df2 = df2.fillna(method='ffill')
regr_line = f'Y = {regr.slope:.2f} * X + {regr.intercept:.2f}'
plt.figure(figsize=(7,7))
plt.plot(df2.X,df2.Y,'x')
plt.plot(df2.X, regr.slope*df2.X+regr.intercept,'r')
plt.legend(['DOW x KOSPI', regr_line])
plt.title(f'DOW x KOSPI (R = {regr.rvalue:.2f})')
plt.xlabel('Dow Jones Industrial Average')
plt.ylabel('KOSPI')
plt.show()
df4 = pd.DataFrame({'A':df['GOLD'],'B':df['KOSPI']})
df4 = df4.fillna(method='bfill')
df4 = df4.fillna(method='ffill')
regr2 = stats.linregress(df4.A,df4.B);regr2
regr_line2 = f'B = {regr2.slope:.2f} * A + {regr2.intercept:.2f}'
plt.figure(figsize=(7,7))
plt.plot(df4.A,df4.B,'.')
plt.plot(df4.A, regr2.slope*df4.A+regr2.intercept,'r')
plt.legend(['GOLD x KOSPI', regr_line2])
plt.title(f'GOLD x KOSPI (R = {regr2.rvalue:.2f})')
plt.xlabel('GOLD PRICE')
plt.ylabel('KOSPI')
plt.show()
df5 = pd.DataFrame({'C':df['USDKRW'],'D':df['KOSPI']})
df5 = df5.fillna(method='bfill')
df5 = df5.fillna(method='ffill')
regr3 = stats.linregress(df5.C,df5.D);regr3
regr_line3 = f'D = {regr3.slope:.2f} * C + {regr3.intercept:.2f}'
plt.figure(figsize=(7,7))
plt.plot(df5.C,df5.D,'X')
plt.plot(df5.C, regr3.slope*df5.C+regr3.intercept,'r')
plt.legend(['USDKRW x KOSPI', regr_line3])
plt.title(f'USDKRW x KOSPI (R = {regr3.rvalue:.2f})')
plt.xlabel('USDKRW')
plt.ylabel('KOSPI')
plt.show()
df1 = df.corr()
fig,ax = plt.subplots(figsize=(10,7))
mask = np.zeros_like(df1,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df1, cmap = 'RdYlBu_r', annot = True, mask=mask, linewidths=.5, cbar_kws={"shrink": .5}, vmin = -1,vmax = 1)
plt.show()
sns.clustermap(df1,annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
df2 = df[-252:].corr()
fig,ax = plt.subplots(figsize=(10,7))
mask = np.zeros_like(df1,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df2, cmap = 'RdYlBu_r', annot = True, mask=mask, linewidths=.5, cbar_kws={"shrink": .5}, vmin = -1,vmax = 1)
plt.show()
sns.clustermap(df2,annot = True, cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
df3 = df[-252:]
# 포트폴리오 구성

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


%matplotlib inline
df = pd.read_excel("C:/it/result_8.xlsx")
close_df = pd.DataFrame({'Date':df['Date'],'kospi_close':df['KOSPI'],'dow_close': df['DOW'],'gold_close':df['GOLD'],'USDKRW_close':df['USDKRW']})
kospi_weight = 0.25
dow_weight = 0.25
gold_weight = 0.25
usd_weight = 0.25

data = close_df[-300:-150] # 2020년 상반기 급락/급등 장
pct_return = (data - data.iloc[0]) / data.iloc[0]
pct_return.columns = ['KOSPI','DOW','GOLD','USDKRW']
kospi_return = pct_return['KOSPI']
dow_return = pct_return['DOW']
gold_return = pct_return['GOLD']
USDKRW_return = pct_return['USDKRW']
portfolio_return = (pct_return * [kospi_weight, dow_weight, gold_weight, usd_weight]).sum(axis=1)

kospi_return.plot(label='KOSPI', figsize=(14,8));
dow_return.plot(label='DOW');
gold_return.plot(label='GOLD');
USDKRW_return.plot(label='USDKRW')
portfolio_return.plot(label='Portfolio');
portfolio_return1.plot(label='Portfolio1')
plt.legend();

# kospi_weight = 0.1
# dow_weight = 0.1
# gold_weight = 0.4
# usd_weight = 0.4