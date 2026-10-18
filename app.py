"""Streamlit app: ประเมินระดับความเสี่ยงโรคระบบทางเดินหายใจตามปัจจัยสิ่งแวดล้อม
ใช้ 2 โมเดล: Random Forest และ XGBoost (เทรนไว้ล่วงหน้าจาก Colab แล้วโหลดจาก model_artifacts/)
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "model_artifacts"

CLASS_LABEL = {0: "ต่ำ (L)", 1: "ปานกลาง (M)", 2: "สูง (H)", 3: "สูงมาก (E)"}
CLASS_COLOR = {0: "#2e9e5b", 1: "#e0a800", 2: "#f07c29", 3: "#d64545"}
CLASS_RULE = {0: "น้อยกว่า 50 ราย/เดือน", 1: "50–100 ราย/เดือน",
              2: "101–150 ราย/เดือน", 3: "มากกว่า 150 ราย/เดือน"}
MONTHS_TH = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

LABELS = {
    "PM25_mean": "PM2.5 เฉลี่ยรายเดือน (µg/m³)",
    "PM25_max": "PM2.5 สูงสุดในเดือน (µg/m³)",
    "AQI_mean": "ระดับ AQI เฉลี่ย (1–5)",
    "Temp_mean": "อุณหภูมิเฉลี่ย (°C)",
    "Humidity_mean": "ความชื้นสัมพัทธ์เฉลี่ย (%)",
    "Pressure_mean": "ความกดอากาศเฉลี่ย (hPa)",
    "Rainfall_sum": "ปริมาณฝนรวมรายเดือน (มม.)",
    "mean_age_month": "อายุเฉลี่ยของผู้ป่วย (ปี)",
    "pct_age_0_4_month": "สัดส่วนผู้ป่วยอายุ 0–4 ปี (%)",
    "pct_age_65plus_month": "สัดส่วนผู้ป่วยอายุ 65 ปีขึ้นไป (%)",
}
BOUNDS = {"PM25_mean": (0.0, 500.0), "PM25_max": (0.0, 700.0), "AQI_mean": (1.0, 5.0),
          "Temp_mean": (-5.0, 50.0), "Humidity_mean": (0.0, 100.0),
          "Pressure_mean": (700.0, 1100.0), "Rainfall_sum": (0.0, 3000.0),
          "mean_age_month": (0.0, 110.0), "pct_age_0_4_month": (0.0, 100.0),
          "pct_age_65plus_month": (0.0, 100.0)}
STEP = {"AQI_mean": 0.05, "Rainfall_sum": 1.0}
ENV_FEATURES = ["PM25_mean", "PM25_max", "AQI_mean", "Temp_mean", "Humidity_mean",
                "Pressure_mean", "Rainfall_sum"]
DEMO_FEATURES = ["mean_age_month", "pct_age_0_4_month", "pct_age_65plus_month"]

st.set_page_config(page_title="ประเมินความเสี่ยงโรคระบบทางเดินหายใจ", page_icon="🫁", layout="wide")


def get_season(d: date) -> str:
    """ฤดูกาลตามเกณฑ์กรมอุตุนิยมวิทยา (เหมือนใน Notebook)"""
    m, day = d.month, d.day
    if (m == 2 and day >= 15) or m in (3, 4) or (m == 5 and day <= 15):
        return "Summer"
    if (m == 5 and day >= 16) or m in (6, 7, 8, 9) or (m == 10 and day <= 15):
        return "Rainy"
    return "Winter"


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_artifacts():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    rf = joblib.load(ART / "random_forest.joblib")
    xgb = joblib.load(ART / "xgboost.joblib")
    profile = pd.read_csv(ART / "province_profile.csv")
    return meta, rf, xgb, profile


def proba4(model, X: pd.DataFrame) -> np.ndarray:
    out = np.zeros(4)
    out[np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)[0]
    return out


def badge(title: str, code: int, conf: float) -> str:
    return (f"<div style='border:1px solid {CLASS_COLOR[code]};border-left:8px solid {CLASS_COLOR[code]};"
            f"border-radius:8px;padding:12px 16px;'>"
            f"<div style='font-size:0.85rem;opacity:.75'>{title}</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{CLASS_COLOR[code]}'>{CLASS_LABEL[code]}</div>"
            f"<div style='font-size:0.85rem;opacity:.75'>ความน่าจะเป็น {conf*100:.1f}% • {CLASS_RULE[code]}</div>"
            f"</div>")


# ------------------------------------------------------------------ โหลดโมเดล
try:
    meta, rf, xgb, profile = load_artifacts()
except FileNotFoundError:
    st.error("ไม่พบโฟลเดอร์ `model_artifacts/` — กรุณารันเซลล์ส่งออกโมเดลใน Colab "
             "(ไฟล์ `colab_export_cell.py`) แล้วนำไฟล์มาวางไว้ข้าง `app.py`")
    st.stop()

FEATS = meta["features"]
STATS = meta["feature_stats"]

st.title("🫁 ประเมินระดับความเสี่ยงโรคระบบทางเดินหายใจ")
st.caption("ตามปัจจัยด้านสิ่งแวดล้อม • Random Forest + XGBoost • ระดับจังหวัด-เดือน")

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("1) เลือกพื้นที่และเดือน")
    provs = profile.sort_values("Province")
    prov = st.selectbox(
        "จังหวัด", provs["Province"].tolist(),
        format_func=lambda p: f"{p} ({provs.loc[provs.Province == p, 'Province_EN'].iloc[0]})")
    month = st.selectbox("เดือน", range(1, 13), format_func=lambda m: MONTHS_TH[m - 1])
    prow = profile[profile["Province"] == prov].iloc[0]
    region, season = prow["Region"], get_season(date(2025, month, 15))
    st.caption(f"ภูมิภาค: **{region}** • ฤดู: **{season}**")
    st.info("ค่าเริ่มต้นของตัวแปรเป็นค่าเฉลี่ยย้อนหลังของจังหวัดที่เลือก "
            "ปรับค่าเพื่อทดลองสถานการณ์ต่าง ๆ ได้")

    hist = {c: float(prow[f"hist_{c}"]) for c in "LMHE" if f"hist_{c}" in prow}
    if hist:
        st.markdown("**ประวัติระดับความเสี่ยงของจังหวัดนี้ (2023–2025)**")
        st.write(" • ".join(f"{k}: {v*100:.0f}%" for k, v in hist.items()))


def num_input(feat: str):
    lo, hi = BOUNDS.get(feat, (STATS[feat]["min"], STATS[feat]["max"]))
    default = float(np.clip(prow[feat], lo, hi))
    step = STEP.get(feat, 0.1)
    return st.number_input(LABELS.get(feat, feat), min_value=float(lo), max_value=float(hi),
                           value=default, step=float(step),
                           format="%.2f" if step < 0.1 else "%.1f",
                           key=f"{prov}|{feat}")


# ------------------------------------------------------------------ Inputs
st.subheader("2) ปัจจัยด้านสิ่งแวดล้อม")
vals = {}
env_present = [f for f in ENV_FEATURES if f in FEATS]
c1, c2 = st.columns(2)
for i, f in enumerate(env_present):
    with (c1 if i % 2 == 0 else c2):
        vals[f] = num_input(f)

demo_present = [f for f in DEMO_FEATURES if f in FEATS]
if demo_present:
    with st.expander("ตัวแปรด้านประชากร (คำนวณจากข้อมูลผู้ป่วย — ดูข้อจำกัดในแท็บ 'ข้อจำกัด')"):
        st.caption("ตัวแปรกลุ่มนี้ได้มาจากข้อมูลผู้ป่วยจริงของเดือนนั้น จึงปกติไม่ทราบล่วงหน้า "
                   "หากไม่ทราบให้ใช้ค่าเริ่มต้น (ค่าเฉลี่ยย้อนหลังของจังหวัด)")
        for f in demo_present:
            vals[f] = num_input(f)

if "PM25_mean" in vals and "PM25_max" in vals and vals["PM25_max"] < vals["PM25_mean"]:
    st.warning("PM2.5 สูงสุดควรมากกว่าหรือเท่ากับค่าเฉลี่ย")

# ------------------------------------------------------------------ Predict
row = {f: 0.0 for f in FEATS}
row.update(vals)
for f in FEATS:
    if f.startswith("Season_"):
        row[f] = int(f == f"Season_{season}")
    elif f.startswith("Region_"):
        row[f] = int(f == f"Region_{region}")
X = pd.DataFrame([row])[FEATS]

p_rf, p_xgb = proba4(rf, X), proba4(xgb, X)
p_avg = (p_rf + p_xgb) / 2

tab_pred, tab_perf, tab_imp, tab_about = st.tabs(
    ["ผลการประเมิน", "ประสิทธิภาพโมเดล", "ความสำคัญของตัวแปร", "ข้อจำกัด"])

with tab_pred:
    k_rf, k_xgb, k_avg = int(p_rf.argmax()), int(p_xgb.argmax()), int(p_avg.argmax())
    a, b, c = st.columns(3)
    a.markdown(badge("Random Forest", k_rf, p_rf[k_rf]), unsafe_allow_html=True)
    b.markdown(badge("XGBoost", k_xgb, p_xgb[k_xgb]), unsafe_allow_html=True)
    c.markdown(badge("ค่าเฉลี่ยของ 2 โมเดล", k_avg, p_avg[k_avg]), unsafe_allow_html=True)

    if k_rf == k_xgb:
        st.success(f"ทั้ง 2 โมเดลให้ผลตรงกัน: {CLASS_LABEL[k_rf]}")
    else:
        st.warning(f"2 โมเดลให้ผลต่างกัน (RF: {CLASS_LABEL[k_rf]}, XGBoost: {CLASS_LABEL[k_xgb]}) "
                   "ควรตีความอย่างระมัดระวัง")

    fig = go.Figure()
    names = [CLASS_LABEL[i] for i in range(4)]
    fig.add_bar(x=names, y=p_rf * 100, name="Random Forest", marker_color="#4c78a8")
    fig.add_bar(x=names, y=p_xgb * 100, name="XGBoost", marker_color="#f58518")
    fig.update_layout(barmode="group", yaxis_title="ความน่าจะเป็น (%)", yaxis_range=[0, 100],
                      height=380, margin=dict(t=30), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("ระดับความเสี่ยงนิยามจากจำนวนผู้ป่วยนอกโรคระบบทางเดินหายใจ (ICD-10 J00–J99) ต่อจังหวัดต่อเดือน: "
               + " | ".join(f"{CLASS_LABEL[i]} = {CLASS_RULE[i]}" for i in range(4)))

with tab_perf:
    st.markdown(f"ประเมินบนชุดทดสอบ {meta['n_test']:,} แถว (ฝึกด้วย {meta['n_train']:,} แถว) — "
                "ค่า Precision / Recall / F1 / AUC เป็น Macro-average")
    st.dataframe(pd.DataFrame(meta["metrics"]).set_index("Model"), width="stretch")
    for name, rep in meta["per_class"].items():
        st.markdown(f"**{name} — รายคลาส**")
        st.dataframe(pd.DataFrame(rep).T.rename(columns={"f1-score": "f1"}).round(3),
                     width="stretch")
    st.caption("คลาส Moderate และ High มีตัวอย่างในชุดทดสอบน้อย ค่า Recall จึงไม่เสถียร")

with tab_imp:
    cols = st.columns(2)
    for col, (name, imp) in zip(cols, meta["importance"].items()):
        s = pd.Series(imp).sort_values().tail(12)
        f = go.Figure(go.Bar(x=s.values, y=[LABELS.get(i, i) for i in s.index], orientation="h"))
        f.update_layout(title=name, height=420, margin=dict(l=10, t=40))
        col.plotly_chart(f, width="stretch")

with tab_about:
    st.markdown("""
**ข้อควรระวังในการตีความ**
- แอปนี้เป็นเครื่องมือประกอบการวิเคราะห์เชิงวิชาการ **ไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์**
- ระดับความเสี่ยงนิยามจากจำนวนผู้ป่วยนอกในข้อมูลที่ได้รับ (ซึ่งอาจมาจากสถานพยาบาลบางแห่ง ไม่ครอบคลุมทุกจังหวัดเท่ากัน)
  โมเดลจึงอาจสะท้อน "ความครอบคลุมของข้อมูลในจังหวัด" ปนกับความเสี่ยงจริง
- ตัวแปรด้านประชากร (อายุเฉลี่ย, สัดส่วนอายุ) คำนวณจากผู้ป่วยในเดือนเดียวกับที่ใช้นิยามระดับความเสี่ยง
  จึงไม่เหมาะกับการ **พยากรณ์ล่วงหน้า** อย่างแท้จริง
- ความกดอากาศเฉลี่ยของแต่ละจังหวัดค่อนข้างคงที่ จึงอาจทำหน้าที่เป็นตัวระบุจังหวัดมากกว่าเป็นสาเหตุทางสิ่งแวดล้อม
- ข้อมูลเป็นรายจังหวัด-เดือน ปี 2023–2025 และชุดทดสอบแบ่งแบบสุ่ม ผลจริงเมื่อใช้กับช่วงเวลาใหม่อาจต่ำกว่าที่รายงาน
""")
