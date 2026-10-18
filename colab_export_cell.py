# =====================================================================
#  เซลล์สำหรับ "ส่งออกโมเดล" — วางต่อท้าย Notebook แล้วรันหลังจากรันเซลล์ก่อนหน้าครบทุกเซลล์
#  เลือกเฉพาะ 2 โมเดลที่ดีที่สุดจากตารางเปรียบเทียบ: Random Forest และ XGBoost
#  ผลลัพธ์: โฟลเดอร์ model_artifacts/ (+ zip ให้ดาวน์โหลด) และ requirements.txt
# =====================================================================
import os, json, shutil
import joblib, sklearn, xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# ตัวแปรที่ใช้ในแอป: ค่าเริ่มต้น = ทั้ง 19 ตัวแปร (ตรงกับตารางเปรียบเทียบในส่วนที่ 3)
# ถ้าต้องการใช้ 12 ตัวแปรสำคัญ ให้เปลี่ยนเป็น list(SELECTED_FEATURES)
APP_FEATURES = list(FEATURE_COLS_FULL)

OUT = "model_artifacts"
os.makedirs(OUT, exist_ok=True)

X_app = model_df[APP_FEATURES]
y_app = model_df["Risk_Level_code"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_app, y_app, test_size=0.2, random_state=RANDOM_STATE, stratify=y_app)

# ---- โมเดลที่ 1: Random Forest (ตั้งค่าเหมือนใน Notebook) ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr, y_tr)

# ---- โมเดลที่ 2: XGBoost (ตั้งค่าเหมือนใน Notebook) ----
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=4, eval_metric="mlogloss",
    random_state=RANDOM_STATE, n_jobs=-1)
xgb.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))

# ---- ประเมินผลบนชุดทดสอบเดียวกัน ----
CLASS_NAMES_APP = ["Low (L)", "Moderate (M)", "High (H)", "Extremely (E)"]
rows, per_class = [], {}
for name, m in [("Random Forest", rf), ("XGBoost", xgb)]:
    pred, proba = m.predict(X_te), m.predict_proba(X_te)
    rows.append(evaluate_model(name, y_te, pred, proba))
    rep = classification_report(y_te, pred, target_names=CLASS_NAMES_APP,
                                zero_division=0, output_dict=True)
    per_class[name] = {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                       for c in CLASS_NAMES_APP}
metrics_df = pd.DataFrame(rows).round(4)
print(metrics_df.to_string(index=False))

# ---- บันทึกโมเดล ----
joblib.dump(rf, f"{OUT}/random_forest.joblib", compress=3)
joblib.dump(xgb, f"{OUT}/xgboost.joblib", compress=3)

# ---- ตารางโปรไฟล์รายจังหวัด (ค่าเฉลี่ยรายเดือนย้อนหลัง; เป็นข้อมูลสรุป ไม่ใช่ข้อมูลผู้ป่วยรายบุคคล) ----
num_feats = [f for f in APP_FEATURES if not f.startswith(("Season_", "Region_"))]
profile = monthly_agg.groupby(["Province", "Region"])[num_feats].mean().reset_index()
TH2EN = {v: k for k, v in EN_TO_TH.items()}
profile["Province_EN"] = profile["Province"].map(TH2EN).fillna(profile["Province"])
dist = (monthly_agg.groupby("Province")["Risk_Level_code"].value_counts(normalize=True)
        .unstack(fill_value=0).reindex(columns=[0, 1, 2, 3], fill_value=0))
dist.columns = ["hist_L", "hist_M", "hist_H", "hist_E"]
profile = profile.merge(dist.reset_index(), on="Province", how="left")
profile.to_csv(f"{OUT}/province_profile.csv", index=False, encoding="utf-8")

# ---- metadata ----
meta = {
    "features": APP_FEATURES,
    "class_names": CLASS_NAMES_APP,
    "feature_stats": {f: {"min": float(X_app[f].min()), "max": float(X_app[f].max()),
                          "median": float(X_app[f].median())} for f in APP_FEATURES},
    "metrics": metrics_df.to_dict("records"),
    "per_class": per_class,
    "importance": {
        "Random Forest": dict(zip(APP_FEATURES, map(float, rf.feature_importances_))),
        "XGBoost": dict(zip(APP_FEATURES, map(float, xgb.feature_importances_))),
    },
    "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
}
with open(f"{OUT}/meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=2)

# ---- requirements.txt ให้เวอร์ชันตรงกับที่ใช้เทรน (กัน error ตอนโหลดโมเดลบน Streamlit) ----
with open("requirements.txt", "w") as fh:
    fh.write("streamlit>=1.50\npandas>=2.0\nnumpy>=1.26\nplotly>=5.20\njoblib>=1.3\n"
             f"scikit-learn=={sklearn.__version__}\nxgboost=={xgboost.__version__}\n")

# ---- ดาวน์โหลด ----
shutil.make_archive("model_artifacts", "zip", OUT)
try:
    from google.colab import files
    files.download("model_artifacts.zip")
    files.download("requirements.txt")
except ImportError:
    pass
print("\nเสร็จสิ้น — นำโฟลเดอร์ model_artifacts/ และ requirements.txt ไปวางในโปรเจกต์ที่มี app.py")
