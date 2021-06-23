from FTAPP.flask_stock_def  import *
"""
venv/FTAPP/
          /static
          /templates
          /flask_stock.py
          /flask_stock_call.py
          /flask_stock_def.py
"""
#
# # String --> JSON
# res = decoding()
# # print(res['score']['kor'])
# print(res['score'][1]['eng'])
#
# # JSON --> String
# res = encoding()
# print(type(res))
#
# # 국고채(1년) 금리 가져오기
# get_api_ecos()
#
# #뉴스 기사글 가져오기
# news_list = xml_to_json()
# for news in news_list:
#     print(news['title'], news['link'])
#
# get_dart_재무재표()

# naver_국내증시()
# naver_craw_시세종합()


# # 모든 회사 목록 가져오기
# allstocks = my_allticker()
# print(allstocks.head())
#
#
# ------------------------------------------
# 코스닥 코스피 코스피200 지수 정보 가져오기
# ------------------------------------------
# import datetime
# import pandas_market_calendars as mcal
# import exchange_calendars as ecals
#
# krkx = ecals.get_calendar('XKRX') #한국 증시 달력
#
# today = datetime.date.today()
# while(True) :
#     if krkx.is_session(today.strftime("%Y%m%d")) == False:  # 오늘은 개장일인지 확인
#         today = today - datetime.timedelta(1)
#         continue
#     else :
#         break
#
# yesterday = today - datetime.timedelta(1)
# while (True):
#     if krkx.is_session(yesterday.strftime("%Y%m%d")) == False:  # 어제가 개장일인지 확인
#         yesterday = yesterday - datetime.timedelta(1)
#         continue
#     else:
#         break
# print(today, yesterday)
# today = today.strftime("%Y%m%d")
# yesterday = yesterday.strftime("%Y%m%d")
# idx_total_list = get_idx_total(yesterday, today)
# print(idx_total_list)
# #
# #
# # #-------------------------------------------
# # # 오늘날짜 기준 등락율 상위 top 50
# # #-------------------------------------------
# df_top50 = get_krx_top50(yesterday, today)
# list_top50 = df_top50.to_json(orient="values")
#
# list_top50 = json.loads(list_top50)
# print(type(list_top50))
#
# print(list_top50)
# # list_top50 = json.dumps(list_top50)
# # print(type(list_top50))
#
# df_list = naver_craw_시세종합('005930')
# html_tab1 = "";
# print(df_list[5])
# print(df_list[7])
# print(df_list[8])
# html_tab3 = ""
# for i, col in enumerate(df_list[5][0].values.tolist()):
#     html_tab3 += col + " : "+ df_list[5][1].values.tolist()[i] +"<br>"
# for i, col in enumerate(df_list[7][0].values.tolist()):
#     html_tab3 += col + " : "+ df_list[7][1].values.tolist()[i] +"<br>"
# for i, col in enumerate(df_list[8][0].values.tolist()):
#     html_tab3 += "<font color='red'><b>"+col + " : "+ df_list[8][1].values.tolist()[i] +"</b></font><br>"
# print(html_tab3)

# df_list = naver_craw_news()
# print(df_list)

list = get_krx_kospi200('20210613','20210615')

# naver_craw('005930')


# html_tab1 = "";
# for i, col in enumerate(df_list[4][0].values.tolist()):
#     html_tab1 += col + " : "+ df_list[4][1].values.tolist()[i] +"<br>"
#
# for i, col in enumerate(df_list[1][0].values.tolist()):
#     html_tab1 += str(col) + " : "+ str(df_list[1][1].values.tolist()[i]) + "<br>"
#
# for i, col in enumerate(df_list[1][2].values.tolist()):
#     html_tab1 += str(col) + ":" + str(df_list[1][3].values.tolist()[i]) + "<br>"
#
# for i, col in enumerate(df_list[0].values.tolist()[0]):
#     html_tab1 += str(i) + ":" + col + "<br>"
# for i, col in enumerate(df_list[0].values.tolist()[1]):
#     html_tab1 += str(i) + ":" + col + "<br>"




#
# #---------------------------------------------------------
# # 네이버 기업 재무재표 크롤링
# #---------------------------------------------------------
# df = naver_craw_시세종합('005930')
# print(df)
