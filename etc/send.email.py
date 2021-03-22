# 1안
import smtplib
from email.message import EmailMessage

message = EmailMessage()
message['Subject']='제목'
message['From']='보낸사람 메일주소'
message['To'] = '받는사람 메일주소'
message.set_content('''내용''')
with smtplib.SMTP_SSL('smtp.naver.com',465) as server: # smtp포트주소, smtp포트넘버
    server.ehlo()
    server.login('아이디', '비밀번호')
    server.send_message(message)

print('이메일을 발송했습니다')
==================================================
# 2안
import smtplib
from email.mime.text import MIMEText
# MIME : 전자우편을 위한 인터넷 표준 포맷
smtp = smtplib.SMTP('smtp.naver.com',465)
smtp.ehlo() # say hello
smtp.starttls() #tls 사용시 필요

smtp.login('아이디','비밀번호')
msg = MIMEText('내용')
msg['Subject']='제목'
msg['From']='보낸사람 메일주소'
msg['To'] = '받는사람 메일주소'
smtp.sendmail('보낸사람메일주소','받는사람 메일주소',msg.as_string())
smtp.quit()
