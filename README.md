# 🏯 智慧旅宿「滯銷風險」預警平台
> Inside Airbnb Taipei × 日系簡約 UI × 淡色系

## 資料
| 檔案 | 說明 | 筆數 |
|------|------|------|
| `data/listings.csv` | 房源基本資料（真實） | 6,241 |
| `data/calendar.csv` | 每日訂房狀態（真實） | 2,277,965 |
| `data/reviews.csv`  | 評論紀錄（真實）     | 210,288 |

## 本機執行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署 Streamlit Community Cloud（免費）
1. 上傳整個資料夾到 GitHub（含 `data/` 子目錄）
2. 前往 https://share.streamlit.io
3. New App → 選 repo → main file: `app.py`
4. Deploy！

> ⚠️ calendar.csv 較大（83MB），GitHub 免費方案限制 100MB，需使用 Git LFS 或上傳較小的子集。

## 風險評分公式
```
滯銷風險分數 = 
  0.40 × (availability_365 / 365)         # 高閒置 → 高風險
  0.35 × (1 - min(reviews, 120) / 120)    # 少評論 → 高風險
  0.15 × (1 - min(rpm, 2.0) / 2.0)        # 低月均評論 → 高風險
  0.10 × (price / price_p95)              # 高價偏貴 → 風險
```
| 分數 | 等級 |
|------|------|
| 0.00 – 0.35 | 🟢 低風險 |
| 0.35 – 0.60 | 🟡 中風險 |
| 0.60 – 1.00 | 🔴 高風險 |
