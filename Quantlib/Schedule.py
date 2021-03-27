# Schedule 이자지급 스케줄을 생성하기 위해 사용
# ql.Schedule(Date effectiveDate, - 효력발생일
#             Date terminationDate, - 만기일 
#             Period tenor, - 이자 지급 주기
#             Calendar calendar,    - 달력
#             BusinessDayConvention convetion,  - 이자결제일
#             BusinessDayConvention terminationDateConvention,  이자결제일의 영업일 관행
#             DateGeneration rule,  날짜생성방식
#             Bool endOfMonth)  월말기준
# ql.Backward - 만기일부터 효력발생일까지 후진방식으로 이자지급 스케줄 생성
# ql.Forward - 효력발생일부터 만기일까지 전진방식으로 이자지급 스케줄 생성
# ql.Zero - 효력발생일과 만기일 사이에 어떠한 결제일도 존재하지 않는다.
# ql.ThirdWednesday  - 효력발생일과 만기일을 제외한 모든 중간 이자지급일을 해당 월의 세번째 수요일로 지정
# ql.Twentieth - 효력발생일을 제외한 모든 이자지급일을 해당 월의 20일로 지정
# ql.TwentiethlMM - 효력발생일을 제외한 모든 이자지급일을 3, 6, 9, 12월 20일로 지정


import QuantLib as ql
# Components
effectiveDate = ql.Date(1,1,2021)
muturityDate = ql.Date(28,3,2025)
tenor = ql.Period(3, ql.Months)
calendar = ql.SouthKorea()
convention = ql.ModifiedFollowing
rule = ql.DateGeneration.Backward
endOfMonth = False

# construction
schedule = ql.Schedule(effectiveDate,maturityDate,tenor,calendar,convetion,convetion,rule,endOfMonth)


ref_date = ql.Date(28,3,2021)

# functions
print("Next Payment Date from {} : {}".format(ref_date, schedule.nextDate(ref_date)))
 # nextDate = 입력받은 날짜 바로 다음에 올 이자지급일
print("Previous Payment Date from {} : {}".format(ref_date, schedule.preivous(ref_date)))
 # priviousDate = 입력받은 날짜 바로 이전에 있었던 이자지급일
