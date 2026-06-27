"""
智慧旅宿滯銷風險預警平台 - 資料清洗腳本
========================================
依據簡報流程：
  ① 原始資料 (listings + reviews + calendar)
  ② 資料清理（價格轉數值、去極端值、評論去雜訊）
  ③ 定義標籤 Y（評論數低 且 空房高）
  ④ 結構化特徵（編碼 / 標準化）
  ⑤ NLP 分支（情緒分數聚合到 listing）
  ⑥ 特徵融合（結構化 ⊕ 文字）

輸出資料夾：cleaned_data/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ───────────────────────────────────────────────
# 0. 建立輸出資料夾
# ───────────────────────────────────────────────
OUTPUT_DIR = "cleaned_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ 輸出資料夾已建立：{OUTPUT_DIR}/")

# ───────────────────────────────────────────────
# 1. 讀取原始資料
# ───────────────────────────────────────────────
print("\n📂 [Step 1] 讀取原始資料...")
listings_raw  = pd.read_csv("listings.csv")
reviews_raw   = pd.read_csv("reviews.csv")
calendar_raw  = pd.read_csv("calendar.csv", low_memory=False)

print(f"  listings  : {listings_raw.shape[0]:,} 筆，{listings_raw.shape[1]} 欄")
print(f"  reviews   : {reviews_raw.shape[0]:,} 筆，{reviews_raw.shape[1]} 欄")
print(f"  calendar  : {calendar_raw.shape[0]:,} 筆，{calendar_raw.shape[1]} 欄")

# ───────────────────────────────────────────────
# 2. 清洗 listings.csv
# ───────────────────────────────────────────────
print("\n🧹 [Step 2] 清洗 listings 資料...")

df = listings_raw.copy()

# 2-1. 移除全部是 NaN 的欄位（neighbourhood_group, license 全空）
always_null = [c for c in df.columns if df[c].isnull().all()]
df.drop(columns=always_null, inplace=True)
print(f"  移除全空欄位：{always_null}")

# 2-2. 處理 price（已是 int，但有 718 筆遺漏 → 用中位數填補）
price_median = df['price'].median()
df['price'] = df['price'].fillna(price_median)
print(f"  price 遺漏值以中位數填補：{price_median:.0f}")

# 2-3. 去除 price 極端值（IQR 法）
Q1, Q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
before = len(df)
df = df[(df['price'] >= lower) & (df['price'] <= upper)]
print(f"  去除 price 極端值（IQR）：{before - len(df)} 筆移除，保留 {len(df):,} 筆")
print(f"  price 合法範圍：{lower:.0f} ~ {upper:.0f}")

# 2-4. 處理 last_review / reviews_per_month（空值 → 從未被評論過）
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['last_review_days_ago'] = (
    pd.Timestamp('2025-12-31') - df['last_review']
).dt.days.fillna(9999).astype(int)  # 從未評論 → 9999 天

# 2-5. 處理 minimum_nights 極端值（>365 天不合理）
df['minimum_nights'] = df['minimum_nights'].clip(upper=365)

# 2-6. 新增衍生特徵
df['is_superhost_proxy'] = (df['calculated_host_listings_count'] >= 3).astype(int)
df['reviews_per_year'] = df['number_of_reviews_ltm']  # 近 12 個月評論數

# 2-7. 重設索引
df = df.reset_index(drop=True)

print(f"  listings 清洗完畢，最終：{df.shape[0]:,} 筆，{df.shape[1]} 欄")

# ───────────────────────────────────────────────
# 3. 從 calendar 計算每個 listing 的空房率
# ───────────────────────────────────────────────
print("\n📅 [Step 3] 計算每個 listing 的年空房率...")

cal = calendar_raw.copy()
cal['available'] = cal['available'].map({'t': 1, 'f': 0})
# 每個 listing 的空房天數 / 總天數
avail_agg = cal.groupby('listing_id')['available'].agg(
    total_days='count',
    avail_days='sum'
).reset_index()
avail_agg['avail_rate'] = avail_agg['avail_days'] / avail_agg['total_days']
avail_agg.rename(columns={'listing_id': 'id'}, inplace=True)
print(f"  calendar 聚合：{avail_agg.shape[0]:,} 個 listing")

# ───────────────────────────────────────────────
# 4. 從 reviews 計算每個 listing 的評論統計
# ───────────────────────────────────────────────
print("\n📝 [Step 4] 計算每個 listing 的評論統計...")

rev = reviews_raw.copy()
rev['date'] = pd.to_datetime(rev['date'], errors='coerce')

# 近 12 個月的評論數
cutoff = pd.Timestamp('2025-12-31') - pd.DateOffset(months=12)
rev_recent = rev[rev['date'] >= cutoff]

review_agg = rev.groupby('listing_id').agg(
    total_reviews=('date', 'count'),
    latest_review=('date', 'max')
).reset_index()

review_recent_agg = rev_recent.groupby('listing_id').agg(
    reviews_last_12m=('date', 'count')
).reset_index()

review_agg = review_agg.merge(review_recent_agg, on='listing_id', how='left')
review_agg['reviews_last_12m'] = review_agg['reviews_last_12m'].fillna(0).astype(int)
review_agg.rename(columns={'listing_id': 'id'}, inplace=True)
print(f"  reviews 聚合：{review_agg.shape[0]:,} 個 listing")

# ───────────────────────────────────────────────
# 5. 定義標籤 Y（滯銷 = 1）
#    規則：近12個月評論數 < 中位數的50% 且 空房率 > 70%
# ───────────────────────────────────────────────
print("\n🏷️  [Step 5] 定義標籤 Y（滯銷與否）...")

# 合併 listings + avail + reviews
df = df.merge(avail_agg[['id', 'avail_rate']], on='id', how='left')
df = df.merge(review_agg[['id', 'total_reviews', 'reviews_last_12m']], on='id', how='left')

# 用 listings 中的 availability_365 補漏（若 calendar 沒有該 listing）
df['avail_rate'] = df['avail_rate'].fillna(df['availability_365'] / 365)
df['reviews_last_12m'] = df['reviews_last_12m'].fillna(df['number_of_reviews_ltm'])
df['total_reviews'] = df['total_reviews'].fillna(df['number_of_reviews'])

# 滯銷標籤定義
review_low_threshold  = df['reviews_last_12m'].quantile(0.33)  # 後三分之一評論少
avail_high_threshold  = 0.70  # 空房率超過 70%

df['is_unsold'] = (
    (df['reviews_last_12m'] <= review_low_threshold) &
    (df['avail_rate'] >= avail_high_threshold)
).astype(int)

print(f"  評論數閾值（低）: <= {review_low_threshold:.1f} 則/年")
print(f"  空房率閾值（高）: >= {avail_high_threshold*100:.0f}%")
print(f"  滯銷 (Y=1)：{df['is_unsold'].sum():,} 筆 ({df['is_unsold'].mean()*100:.1f}%)")
print(f"  正常 (Y=0)：{(1-df['is_unsold']).sum():,} 筆 ({(1-df['is_unsold']).mean()*100:.1f}%)")

# ───────────────────────────────────────────────
# 6. 結構化特徵工程（編碼 + 標準化）
# ───────────────────────────────────────────────
print("\n⚙️  [Step 6] 結構化特徵工程...")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df_feat = df.copy()

# 6-1. 類別欄位 Label Encoding
cat_cols = ['room_type', 'neighbourhood']
for col in cat_cols:
    le = LabelEncoder()
    df_feat[col + '_enc'] = le.fit_transform(df_feat[col].astype(str))
    print(f"  LabelEncoded: {col} → {col}_enc ({df_feat[col].nunique()} 類別)")

# 6-2. 選取結構化特徵欄位（⚠️ 不放 Y 相關欄位）
structured_features = [
    'id',
    'price',
    'minimum_nights',
    'calculated_host_listings_count',
    'latitude',
    'longitude',
    'last_review_days_ago',
    'reviews_per_month',
    'is_superhost_proxy',
    'avail_rate',           # 來自 calendar（用於 Y 定義，僅保留於特徵版）
    'room_type_enc',
    'neighbourhood_enc',
]

df_struct = df_feat[structured_features + ['is_unsold']].copy()

# 6-3. 數值欄位 MinMax 標準化（不含 id, is_unsold）
num_cols = [c for c in structured_features if c not in ('id',)]
scaler = MinMaxScaler()
df_struct_scaled = df_struct.copy()
df_struct_scaled[num_cols] = scaler.fit_transform(df_struct[num_cols])

print(f"  結構化特徵完成：{len(structured_features)} 個特徵欄")

# ───────────────────────────────────────────────
# 7. NLP 情緒模擬特徵（無評論文字時用代理指標）
#    實際專案應用 TextBlob / VADER / Sentence-BERT
# ───────────────────────────────────────────────
print("\n💬 [Step 7] 建立 NLP 情緒代理特徵...")

# 因 reviews.csv 僅含 listing_id + date，無評論文字
# → 用「評論頻率」、「評論新舊度」作為情緒代理指標
df_nlp = df[['id']].copy()

# 近期活躍度（近12月評論 / 總評論）→ 代理「正面情緒動能」
df_nlp['review_recency_ratio'] = np.where(
    df['total_reviews'] > 0,
    df['reviews_last_12m'] / df['total_reviews'],
    0
)

# 評論衰退指標（距離最後評論天數，標準化）
max_days = df['last_review_days_ago'].replace(9999, np.nan).max()
df_nlp['review_decay'] = np.where(
    df['last_review_days_ago'] == 9999,
    1.0,  # 從未評論 → 最大衰退
    df['last_review_days_ago'] / max_days
)

# 評論密度（每年評論數 / listing 均值）
mean_reviews = df['reviews_last_12m'].mean()
df_nlp['review_density_ratio'] = df['reviews_last_12m'] / (mean_reviews + 1e-6)
df_nlp['review_density_ratio'] = df_nlp['review_density_ratio'].clip(0, 5)  # clip 極端值

# 合併 is_unsold
df_nlp['is_unsold'] = df['is_unsold'].values

print(f"  NLP代理特徵完成：review_recency_ratio, review_decay, review_density_ratio")

# ───────────────────────────────────────────────
# 8. 特徵融合（結構化 ⊕ NLP）
# ───────────────────────────────────────────────
print("\n🔗 [Step 8] 特徵融合（結構化 ⊕ NLP代理）...")

nlp_cols_to_merge = ['id', 'review_recency_ratio', 'review_decay', 'review_density_ratio']
df_merged = df_struct_scaled.merge(df_nlp[nlp_cols_to_merge], on='id', how='left')
print(f"  融合後特徵矩陣：{df_merged.shape[0]:,} 筆，{df_merged.shape[1]} 欄")

# ───────────────────────────────────────────────
# 9. 儲存所有清洗後的 CSV 檔案
# ───────────────────────────────────────────────
print(f"\n💾 [Step 9] 儲存 CSV 至 {OUTPUT_DIR}/...")

# 9-1. 清洗後的 listings（含 Y 標籤，含原始欄位）
output_listings = df[[
    'id', 'name', 'host_id', 'host_name', 'neighbourhood', 'room_type',
    'latitude', 'longitude',
    'price', 'minimum_nights',
    'number_of_reviews', 'reviews_per_month', 'last_review',
    'last_review_days_ago',
    'calculated_host_listings_count', 'availability_365',
    'number_of_reviews_ltm', 'reviews_last_12m', 'total_reviews',
    'avail_rate', 'is_superhost_proxy',
    'is_unsold'
]].copy()
output_listings.to_csv(f"{OUTPUT_DIR}/01_listings_cleaned.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 01_listings_cleaned.csv  →  {output_listings.shape}")

# 9-2. 結構化特徵（原始尺度）
df_struct.to_csv(f"{OUTPUT_DIR}/02_structured_features.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 02_structured_features.csv  →  {df_struct.shape}")

# 9-3. 結構化特徵（標準化）
df_struct_scaled.to_csv(f"{OUTPUT_DIR}/03_structured_features_scaled.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 03_structured_features_scaled.csv  →  {df_struct_scaled.shape}")

# 9-4. NLP 情緒代理特徵
df_nlp.to_csv(f"{OUTPUT_DIR}/04_nlp_proxy_features.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 04_nlp_proxy_features.csv  →  {df_nlp.shape}")

# 9-5. 融合特徵矩陣（模型訓練用）
df_merged.to_csv(f"{OUTPUT_DIR}/05_merged_features_for_model.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 05_merged_features_for_model.csv  →  {df_merged.shape}")

# 9-6. Calendar 聚合（空房率）
avail_agg.to_csv(f"{OUTPUT_DIR}/06_calendar_avail_rate.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 06_calendar_avail_rate.csv  →  {avail_agg.shape}")

# 9-7. Reviews 聚合（評論統計）
review_agg.to_csv(f"{OUTPUT_DIR}/07_reviews_aggregated.csv", index=False, encoding='utf-8-sig')
print(f"  ✅ 07_reviews_aggregated.csv  →  {review_agg.shape}")

# ───────────────────────────────────────────────
# 10. 摘要報告
# ───────────────────────────────────────────────
print("\n" + "="*60)
print("📊 資料清洗摘要報告")
print("="*60)
print(f"原始 listings 筆數：{listings_raw.shape[0]:,}")
print(f"清洗後 listings 筆數：{df.shape[0]:,}（移除 {listings_raw.shape[0]-df.shape[0]:,} 筆極端值）")
print(f"原始 reviews 筆數：{reviews_raw.shape[0]:,}")
print(f"calendar 涵蓋 listings 數：{avail_agg.shape[0]:,}")
print(f"\n標籤分佈：")
print(f"  滯銷 (is_unsold=1)：{df['is_unsold'].sum():,} 筆 ({df['is_unsold'].mean()*100:.1f}%)")
print(f"  正常 (is_unsold=0)：{(df['is_unsold']==0).sum():,} 筆 ({(df['is_unsold']==0).mean()*100:.1f}%)")
print(f"\n輸出檔案位置：{os.path.abspath(OUTPUT_DIR)}/")
print(f"  01_listings_cleaned.csv         - 清洗後完整 listings 含標籤")
print(f"  02_structured_features.csv      - 結構化特徵（原始尺度）")
print(f"  03_structured_features_scaled.csv - 結構化特徵（MinMax 標準化）")
print(f"  04_nlp_proxy_features.csv       - NLP 情緒代理特徵")
print(f"  05_merged_features_for_model.csv - 融合特徵矩陣（模型訓練用）")
print(f"  06_calendar_avail_rate.csv      - Calendar 空房率聚合")
print(f"  07_reviews_aggregated.csv       - Reviews 評論統計聚合")
print("="*60)
print("✅ 全部完成！")
