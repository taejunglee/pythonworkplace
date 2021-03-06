def getCAGR(first, last, years): # 주가, 연도
    return (last/first)**(1/years)-1

cagr = getCAGR( , , ) # 주가, 연도
print("SEC CAGR : {:.2%}".format(cagr))
