import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# [수정됨] datasets 폴더 경로 반영
csv_file = "datasets/Cyberpunk_2077_Steam_Reviews.csv" 
print(f"Loading {csv_file}...")

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"원본 데이터가 없습니다: {csv_file}")

df = pd.read_csv(csv_file, usecols=['Rating', 'Date Posted'])

print("Processing data...")
df['Date Posted'] = pd.to_datetime(df['Date Posted'], format='%m/%d/%Y', errors='coerce')
df['Is_Positive'] = df['Rating'].apply(lambda x: 1 if x == 'Recommended' else 0)
df = df.dropna(subset=['Date Posted'])

daily_sentiment = df.groupby('Date Posted')['Is_Positive'].mean().reset_index()
daily_sentiment.columns = ['Date', 'Positive_Ratio']
daily_sentiment = daily_sentiment.sort_values('Date')
daily_sentiment['Smoothed_Ratio'] = daily_sentiment['Positive_Ratio'].rolling(window=7, min_periods=1).mean()

# 그래프 저장 [수정됨]
output_img = "png/ground_truth_steam.png"
if not os.path.exists("png"):
    os.makedirs("png")

plt.figure(figsize=(15, 6))
sns.set_theme(style="whitegrid")
plt.plot(daily_sentiment['Date'], daily_sentiment['Smoothed_Ratio'], linestyle='-', linewidth=2, color='#00a8cc', label='7-Day Moving Avg')
plt.title('Cyberpunk 2077 Steam Review Sentiment Trend', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Positive Review Ratio')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig(output_img, dpi=300)
print(f"Graph saved to '{output_img}'")

# CSV 저장 [수정됨]
daily_sentiment.to_csv("datasets/ground_truth_steam.csv", index=False)
print("'datasets/ground_truth_steam.csv' 저장 완료")

