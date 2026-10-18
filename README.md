# 🫁 ประเมินความเสี่ยงโรคระบบทางเดินหายใจตามปัจจัยสิ่งแวดล้อม

แอป Streamlit ใช้ 2 โมเดลที่ดีที่สุดจาก Notebook: **Random Forest** และ **XGBoost**

## ขั้นตอน
1. เปิด Notebook ใน Colab รันทุกเซลล์ให้ครบ
2. วางโค้ดจาก `colab_export_cell.py` เป็นเซลล์ใหม่ท้ายสุด แล้วรัน (จะดาวน์โหลด `model_artifacts.zip` และ `requirements.txt`)
3. แตก zip วางเป็นโฟลเดอร์ `model_artifacts/` ข้าง `app.py` และใช้ `requirements.txt` ที่ดาวน์โหลดมาแทนไฟล์ตัวอย่าง
4. ทดสอบ: `pip install -r requirements.txt && streamlit run app.py`
5. push ขึ้น GitHub แล้ว Deploy ที่ https://share.streamlit.io (Main file: `app.py`)

> ห้ามอัปโหลดไฟล์ข้อมูลดิบ (.xlsx) ขึ้น GitHub — โปรเจกต์นี้ใช้เฉพาะโมเดลและตารางค่าเฉลี่ยรายจังหวัด

## โครงสร้าง
```
app.py
requirements.txt
model_artifacts/
  random_forest.joblib
  xgboost.joblib
  meta.json
  province_profile.csv
```
