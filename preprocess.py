"""
preprocess.py
─────────────────────────────────────────────
在本機執行一次，將大型 CSV 壓縮成小型彙整檔，
方便上傳 GitHub 並部署至 Streamlit Community Cloud。

Usage:
  python preprocess.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data"

print("【1/2】 處理 calendar.csv（~83 MB）…")
cal = pd.read_csv(DATA / "calendar.csv",
                  usecols=["date", "available"], low_memory=False)
cal["date"]      = pd.to_datetime(cal["date"])
cal["month"]     = cal["date"].dt.to_period("M").astype(str)
cal["available"] = (cal["available"] == "t").astype(int)
mc = (cal.groupby("month")["available"]
         .agg(avg_availability="mean", total_slots="count")
         .reset_index())
mc["avg_occupancy_pct"]    = ((1 - mc["avg_availability"]) * 100).round(2)
mc["avg_availability_pct"] = (mc["avg_availability"] * 100).round(2)
mc.to_csv(DATA / "monthly_calendar.csv", index=False, encoding="utf-8")
print(f"   ✅ monthly_calendar.csv → {len(mc)} 列 "
      f"（{(DATA/'monthly_calendar.csv').stat().st_size//1024} KB）")

print("【2/2】 處理 reviews.csv…")
rev = pd.read_csv(DATA / "reviews.csv",
                  usecols=["date"], low_memory=False)
rev["date"]  = pd.to_datetime(rev["date"], errors="coerce")
rev["month"] = rev["date"].dt.to_period("M").astype(str)
mr = rev.groupby("month").size().reset_index(name="review_count")
mr.to_csv(DATA / "monthly_reviews.csv", index=False, encoding="utf-8")
print(f"   ✅ monthly_reviews.csv  → {len(mr)} 列 "
      f"（{(DATA/'monthly_reviews.csv').stat().st_size//1024} KB）")

print("\n完成！上傳至 GitHub 時，data/ 只需保留：")
print("  ・listings.csv")
print("  ・monthly_calendar.csv")
print("  ・monthly_reviews.csv")
print("（calendar.csv / reviews.csv 太大，可加入 .gitignore）")
