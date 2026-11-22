# -*- coding: utf-8 -*-
"""Demo Streamlit: dự đoán khách hàng đăng ký tiền gửi kỳ hạn (v2).

Thay đổi v2 (sau góp ý 2026-11-14):
- Dùng mô hình ĐÃ CALIBRATE (models/model_calibrated.joblib) nên con số hiển
  thị là xác suất trung thực, kèm phương pháp calibrate trong metadata.
  Nếu chưa có mô hình calibrate thì dùng model.joblib và đổi nhãn thành
  "Điểm ưu tiên" (không gọi là xác suất).
- Ngưỡng chính chọn theo chi phí-lợi ích trên tập validation theo thời gian.
- Upload CSV: kiểm tra cột bắt buộc, kiểu số, báo lỗi dễ hiểu; cảnh báo khi
  campaign > 15 bị cắt trần; cảnh báo category chưa từng thấy.
- Giải thích từng dự đoán: top yếu tố đóng góp bằng SHAP (mô hình cây) hoặc
  hệ số tuyến tính (LogReg).
- Mô hình KHÔNG dùng biến duration (chỉ biết sau khi gọi).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Mô hình đã lưu (pickle) tham chiếu tới `src.features`, nhưng khi chạy
# `streamlit run app/app.py`, Streamlit chỉ đưa thư mục app/ vào sys.path
# nên `joblib.load` lỗi "No module named 'src'". Thêm thư mục gốc dự án
# vào sys.path để app chạy được bằng đúng lệnh trong README (và trong Docker).
sys.path.insert(0, str(ROOT))
CAL_MODEL_PATH = ROOT / "models" / "model_calibrated.joblib"
MODEL_PATH = ROOT / "models" / "model.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

RAW_COLUMNS = ["age", "job", "marital", "education", "default", "housing",
               "loan", "contact", "month", "day_of_week", "duration",
               "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
               "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUMERIC_COLUMNS = ["age", "campaign", "pdays", "previous", "duration",
                   "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                   "euribor3m", "nr.employed"]
CAT_COLUMNS = [c for c in RAW_COLUMNS if c not in NUMERIC_COLUMNS]
CAMPAIGN_CAP = 15

JOB = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
       "retired", "self-employed", "services", "student", "technician",
       "unemployed", "unknown"]
MARITAL = ["divorced", "married", "single", "unknown"]
EDUCATION = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
             "professional.course", "university.degree", "unknown"]
YES_NO_UNKNOWN = ["no", "yes", "unknown"]
YES_NO = ["no", "yes"]
CONTACT = ["cellular", "telephone"]
MONTH = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DOW = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME = ["failure", "nonexistent", "success"]

MACRO_DEFAULT = {"emp.var.rate": -0.1, "cons.price.idx": 93.994,
                 "cons.conf.idx": -36.4, "euribor3m": 1.33, "nr.employed": 5191.0}


@st.cache_resource
def load_model():
    """Trả về (model, threshold_dict, calibrated: bool, metadata: dict|None)."""
    thr = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    meta = None
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if CAL_MODEL_PATH.exists():
        return joblib.load(CAL_MODEL_PATH), thr, True, meta
    return joblib.load(MODEL_PATH), thr, False, meta


def base_pipeline(model):
    """Bóc lớp CalibratedClassifierCV để lấy pipeline gốc (cho SHAP)."""
    if hasattr(model, "calibrated_classifiers_"):
        est = model.calibrated_classifiers_[0].estimator
        # sklearn 1.9: mô hình prefit được bọc trong FrozenEstimator
        if est.__class__.__name__ == "FrozenEstimator":
            est = est.estimator
        return est
    return model


def explain_prediction(model, df_row):
    """Top 5 yếu tố đóng góp cho một dự đoán (SHAP nếu là mô hình cây,
    nếu không thì hệ số tuyến tính). Trả về list (tên, giá trị) hoặc None."""
    try:
        import shap
        pipe = base_pipeline(model)
        clf = pipe.named_steps["clf"]
        x_fe = pipe.named_steps["fe"].transform(df_row)
        x_prep = pipe.named_steps["prep"].transform(x_fe)
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        if hasattr(clf, "estimators_") or clf.__class__.__name__ in (
                "RandomForestClassifier", "HistGradientBoostingClassifier",
                "LGBMClassifier", "DecisionTreeClassifier", "XGBClassifier"):
            expl = shap.TreeExplainer(clf)
            vals = expl.shap_values(x_prep)
            if isinstance(vals, list):
                vals = vals[1]
            elif getattr(vals, "ndim", 2) == 3:
                vals = vals[:, :, 1]
            row_vals = np.asarray(vals)[0]
        else:  # mô hình tuyến tính: đóng góp = hệ số * giá trị đã chuẩn hóa
            row_vals = clf.coef_[0] * np.asarray(x_prep)[0]
        order = np.argsort(-np.abs(row_vals))[:5]
        return [(names[i].replace("num__", "").replace("cat__", ""), float(row_vals[i]))
                for i in order]
    except Exception:
        return None


def validate_upload(data):
    """Kiểm tra CSV tải lên. Trả về (data_đã_lọc, danh_sách_cảnh_báo, lỗi)."""
    warnings, errors = [], []
    missing = [c for c in RAW_COLUMNS if c not in data.columns]
    if missing:
        errors.append(f"Thiếu cột bắt buộc: {', '.join(missing)}. "
                      "File cần đủ 20 cột như dữ liệu gốc (kể cả duration, "
                      "dù mô hình không dùng nó).")
        return None, warnings, errors
    data = data[list(RAW_COLUMNS)].copy()
    # Kiểu số: ép kiểu, dòng nào lỗi thì báo rõ và loại ra
    for col in NUMERIC_COLUMNS:
        bad = pd.to_numeric(data[col], errors="coerce").isna() & data[col].notna()
        if bad.any():
            warnings.append(f"Cột {col}: {int(bad.sum())} dòng không phải số "
                            "(đã loại khỏi danh sách).")
            data = data[~bad]
        data[col] = pd.to_numeric(data[col], errors="coerce")
    before = len(data)
    data = data.dropna(subset=NUMERIC_COLUMNS)
    if len(data) < before:
        warnings.append(f"Đã loại {before - len(data)} dòng thiếu giá trị số.")
    if len(data) == 0:
        errors.append("Không còn dòng hợp lệ nào sau khi kiểm tra. "
                      "Hãy xem lại định dạng số trong file (dùng dấu chấm thập phân).")
        return None, warnings, errors
    # Cảnh báo campaign bị cắt trần + category lạ
    n_clip = int((data["campaign"] > CAMPAIGN_CAP).sum())
    if n_clip:
        warnings.append(f"{n_clip} dòng có campaign > {CAMPAIGN_CAP} sẽ bị cắt "
                        f"trần ở {CAMPAIGN_CAP} như lúc huấn luyện.")
    for col, known in [("job", JOB), ("marital", MARITAL), ("education", EDUCATION),
                       ("contact", CONTACT), ("month", MONTH), ("day_of_week", DOW),
                       ("poutcome", POUTCOME)]:
        unseen = sorted(set(data[col].astype(str)) - set(known))
        if unseen:
            warnings.append(f"Cột {col}: giá trị chưa từng thấy khi huấn luyện "
                            f"({', '.join(unseen[:5])}) — mô hình vẫn chạy, "
                            "coi như category lạ.")
    return data, warnings, errors


def single_input_form():
    st.subheader("Nhập thông tin một khách hàng")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Tuổi", 17, 99, 40)
        job = st.selectbox("Nghề nghiệp", JOB)
        marital = st.selectbox("Hôn nhân", MARITAL)
        education = st.selectbox("Học vấn", EDUCATION)
        default = st.selectbox("Nợ xấu (default)", YES_NO_UNKNOWN)
        housing = st.selectbox("Vay mua nhà", YES_NO_UNKNOWN)
        loan = st.selectbox("Vay cá nhân", YES_NO_UNKNOWN)
    with c2:
        contact = st.selectbox("Kênh liên hệ", CONTACT)
        month = st.selectbox("Tháng liên hệ", MONTH, index=4)
        day_of_week = st.selectbox("Thứ liên hệ", DOW)
        campaign = st.number_input("Số lần gọi trong chiến dịch này", 1, 60, 1)
        if campaign > CAMPAIGN_CAP:
            st.warning(f"Campaign > {CAMPAIGN_CAP} sẽ bị cắt trần ở {CAMPAIGN_CAP} "
                       "như lúc huấn luyện.")
        pdays = st.number_input("Số ngày từ lần liên hệ trước (999 = chưa từng)", 0, 999, 999)
        previous = st.number_input("Số lần liên hệ trước chiến dịch", 0, 10, 0)
        poutcome = st.selectbox("Kết quả chiến dịch trước", POUTCOME, index=1)
    with c3:
        st.caption("Chỉ số vĩ mô (mặc định = trung vị tập train)")
        macro = {k: st.number_input(k, value=float(v))
                 for k, v in MACRO_DEFAULT.items()}
    row = {"age": age, "job": job, "marital": marital, "education": education,
           "default": default, "housing": housing, "loan": loan,
           "contact": contact, "month": month, "day_of_week": day_of_week,
           "duration": 0, "campaign": campaign, "pdays": pdays,
           "previous": previous, "poutcome": poutcome, **macro}
    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title="Bank Marketing - Dự đoán đăng ký tiền gửi", layout="wide")
    st.title("Dự đoán khách hàng đăng ký tiền gửi kỳ hạn")
    st.write("Mô hình xếp hạng khách hàng theo xác suất đồng ý, giúp ngân hàng ưu tiên "
             "gọi nhóm đầu bảng. Mô hình **không dùng biến duration** (chỉ biết sau khi gọi).")
    try:
        model, thr, calibrated, meta = load_model()
    except FileNotFoundError:
        st.error("Chưa có mô hình. Hãy chạy scripts/08b_refit80.py trước.")
        return
    thr_main = thr.get("threshold_cost_benefit", thr.get("threshold_f1"))
    label = "Xác suất khách đăng ký (đã hiệu chỉnh)" if calibrated else "Điểm ưu tiên"
    calib = (meta or {}).get("calibration_method", "?")
    thr_src = thr.get("threshold_source", "OOF train80")
    st.info(f"Ngưỡng quyết định (chi phí-lợi ích, chọn trên {thr_src}): "
            f"**{thr_main:.3f}** | Ngưỡng tối ưu F1: **{thr.get('threshold_f1', float('nan')):.3f}**"
            + (f" | Xác suất đã calibrate ({calib} — xem metadata)." if calibrated
               else " | Mô hình chưa calibrate nên hiển thị điểm ưu tiên, không phải xác suất."))
    if meta:
        with st.expander("Thông tin mô hình"):
            st.json(meta)

    tab1, tab2 = st.tabs(["Một khách hàng", "Xếp hạng hàng loạt (CSV)"])
    with tab1:
        df = single_input_form()
        if st.button("Dự đoán", type="primary"):
            proba = float(model.predict_proba(df)[0, 1])
            st.metric(label, f"{proba:.1%}")
            if proba >= thr_main:
                st.success("Nên gọi: xác suất trên ngưỡng quyết định.")
            else:
                st.warning("Chưa nên ưu tiên gọi: xác suất dưới ngưỡng quyết định.")
            st.progress(min(proba, 1.0))
            expl = explain_prediction(model, df)
            if expl:
                st.subheader("Vì sao mô hình chấm điểm này? (top yếu tố đóng góp)")
                for name, val in expl:
                    st.write(f"- **{name}**: {'+' if val >= 0 else ''}{val:.3f}")
    with tab2:
        st.write("Tải lên file CSV có đúng các cột như dữ liệu gốc (không cần cột y). "
                 "Hệ thống trả về danh sách xếp hạng theo xác suất.")
        up = st.file_uploader("Chọn file CSV (phân tách bằng ';')", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up, sep=";")
            except Exception as e:
                st.error(f"Không đọc được file CSV: {e}. File phải phân tách bằng dấu ';'.")
                return
            data, warnings, errors = validate_upload(raw)
            for w in warnings:
                st.warning(w)
            if errors:
                for e in errors:
                    st.error(e)
                return
            data["proba"] = model.predict_proba(data)[:, 1]
            data = data.sort_values("proba", ascending=False).reset_index(drop=True)
            data["rank"] = data.index + 1
            data["goi"] = (data["proba"] >= thr_main).map({True: "Gọi", False: "Chưa gọi"})
            st.dataframe(data)
            st.download_button("Tải danh sách đã xếp hạng",
                               data.to_csv(sep=";", index=False).encode("utf-8"),
                               "ranked_customers.csv", "text/csv")


if __name__ == "__main__":
    main()
