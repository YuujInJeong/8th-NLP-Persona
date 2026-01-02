import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# [수정됨] datasets 폴더 경로 반영
steam_csv = "datasets/ground_truth_steam.csv"
print(f"{steam_csv} 로드 완료")

# 1. 스팀 데이터 로드 (날짜 범위 확인용)
if not os.path.exists(steam_csv):
    raise FileNotFoundError(f"파일이 없습니다: {steam_csv}. analyze_ground_truth_steam.py 먼저 실행하세요.")

df_steam = pd.read_csv(steam_csv)
df_steam['Date'] = pd.to_datetime(df_steam['Date'])

min_date = df_steam['Date'].min()
max_date = df_steam['Date'].max()

start_date_str = min_date.strftime('%Y-%m-%d')
end_date_str = max_date.strftime('%Y-%m-%d')

print(f"기간 설정: {start_date_str} ~ {end_date_str}")

# 2. 주가 데이터 다운로드
TICKER = "CDR.WA"
print(f"{TICKER} 주가 데이터 다운로드 중...")
stock_data = yf.download(TICKER, start=start_date_str, end=end_date_str)

df = stock_data[['Close']].reset_index()
df.columns = ['Date', 'Stock_Price']
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

# 4. CSV 저장 [수정됨]
output_csv = "datasets/ground_truth_stock.csv"
df.to_csv(output_csv, index=False)
print(f"주가 데이터 저장 완료: '{output_csv}'")

# 5. 그래프 저장 [수정됨] -> png 폴더
output_img = "png/ground_truth_stock.png"
# png 폴더 없으면 생성
if not os.path.exists("png"):
    os.makedirs("png")

plt.figure(figsize=(15, 6))
sns.set_theme(style="whitegrid")
plt.plot(df['Date'], df['Stock_Price'], linestyle='-', linewidth=2, color='#e63946', label=f'CD Projekt S.A. ({TICKER})')
plt.xlim(df['Date'].min(), df['Date'].max())
plt.title('CD Projekt Red Market Sentiment Trend(stock price)', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(output_img, dpi=300)
print(f"'{output_img}'로 저장 완료!")

