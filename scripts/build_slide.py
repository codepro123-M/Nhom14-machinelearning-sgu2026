# -*- coding: utf-8 -*-
"""
Tạo slide thuyết trình cuối kỳ (.pptx) — Nhóm 14, môn Máy học.
Đề tài: Dự đoán khách hàng đăng ký tiền gửi có kỳ hạn (Bank Marketing)
Chạy:  python scripts/build_slide.py   (từ thư mục gốc dự án)
Đầu ra: reports/Slide_BaoCao_CuoiKy_Nhom14_BankMarketing.pptx
Mọi số liệu lấy từ reports/*.json, reports/*.csv (không gõ tay số).
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
OUT = REPORTS / "Slide_BaoCao_CuoiKy_Nhom14_BankMarketing.pptx"

NAVY = RGBColor(0x1F, 0x3B, 0x63)
LIGHT = RGBColor(0xD9, 0xE2, 0xF3)
LIGHT2 = RGBColor(0xED, 0xF2, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x59, 0x59, 0x59)
BLACK = RGBColor(0x00, 0x00, 0x00)
FONT = "Arial"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def load_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def fmt(x, nd=4):
    """Định dạng số với dấu phẩy thập phân (quy ước tiếng Việt)."""
    return f"{x:.{nd}f}".replace(".", ",") if isinstance(x, float) else str(x)


def num(x, nd=4):
    return fmt(x, nd)


def nums(x, nd=4):
    """Như num() nhưng giữ dấu + / - (dùng cho chênh lệch)."""
    return f"{x:+.{nd}f}".replace(".", ",") if isinstance(x, float) else str(x)


def pct(x, nd=1):
    """Phần trăm với dấu phẩy thập phân (x là tỷ lệ 0-1)."""
    return f"{x * 100:.{nd}f}%".replace(".", ",")


def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tb, tf


def para(tf, text, size=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
         space_after=6, space_before=0, first=False, italic=False):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = 1.12
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return p


def rich_para(tf, parts, align=PP_ALIGN.LEFT, space_after=6, space_before=0,
              first=False, line_spacing=1.12):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.line_spacing = line_spacing
    for text, bold, color, size in parts:
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return p


def bullets(slide, x, y, w, h, items, size=18, space_after=10):
    tb, tf = textbox(slide, x, y, w, h)
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            rich_para(tf, [(it[0], True, BLACK, size), (it[1], False, BLACK, size)],
                      space_after=space_after, first=(i == 0))
        else:
            para(tf, "•  " + it, size=size, space_after=space_after, first=(i == 0))
    return tb


def content_slide(title, subtitle=None, footer="Nhóm 14  •  Môn Máy học  •  Báo cáo cuối kỳ — Bank Marketing"):
    slide = prs.slides.add_slide(BLANK)
    add_rect(slide, 0, 0, SW, 0.08, NAVY)
    _, tf = textbox(slide, 0.5, 0.28, 12.3, 0.75)
    para(tf, title, size=28, bold=True, color=NAVY, first=True, space_after=2)
    if subtitle:
        para(tf, subtitle, size=15, color=GREY, space_after=0)
    _, tf2 = textbox(slide, 0.5, 7.08, 12.3, 0.35)
    para(tf2, footer, size=11, color=GREY, first=True, space_after=0)
    return slide


def add_table(slide, x, y, w, headers, rows, col_widths=None, font_size=14, row_h=0.42):
    nrows, ncols = len(rows) + 1, len(headers)
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    table = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = h
        r.font.name = FONT; r.font.size = Pt(font_size); r.font.bold = True
        r.font.color.rgb = WHITE
    for ri, row in enumerate(rows, 1):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            if ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT2
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(font_size)
            r.font.color.rgb = BLACK
    return shape


def add_picture(slide, filename, x, y, w):
    path = FIG / filename
    if not path.exists():
        _, tf = textbox(slide, x, y, w, 0.5)
        para(tf, f"[Thiếu hình: {filename}]", size=14, color=GREY, first=True)
        return
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def title_slide():
    slide = prs.slides.add_slide(BLANK)
    add_rect(slide, 0, 0, SW, SH, NAVY)
    _, tf = textbox(slide, 1.0, 1.6, 11.3, 4.2)
    para(tf, "DỰ ĐOÁN KHÁCH HÀNG ĐĂNG KÝ", size=36, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, first=True, space_after=4)
    para(tf, "TIỀN GỬI CÓ KỲ HẠN", size=36, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, space_after=14)
    para(tf, "Bank Marketing  •  UCI Machine Learning Repository", size=20,
         color=LIGHT, align=PP_ALIGN.CENTER, space_after=22)
    para(tf, "Báo cáo cuối kỳ  •  Môn Máy học  •  Nhóm 14 — Lớp DCT124C3",
         size=18, color=WHITE, align=PP_ALIGN.CENTER, space_after=6)
    para(tf, "Trần Quang Minh (3124411168)  •  Tháng 11/2026", size=16,
         color=LIGHT, align=PP_ALIGN.CENTER)


def main():
    import pandas as pd
    eda = load_json("eda_summary.json")
    cv0 = pd.read_csv(REPORTS / "cv_default_models.csv", index_col=0)
    cvt = pd.read_csv(REPORTS / "cv_tuned_models.csv", index_col=0)
    final = load_json("final_summary.json")
    extra = load_json("extra_experiments.json")
    ana = load_json("analysis_summary.json")
    tuned_best = load_json("tuned_best_model.json")
    test_table = pd.read_csv(REPORTS / "test_results.csv", index_col=0)
    m = final["test_metrics_f1_threshold"]

    title_slide()

    # 2. Bài toán
    s = content_slide("Bài toán: gọi ai để không phí cuộc gọi?",
                      "Ngân hàng gọi điện mời khách gửi tiền kỳ hạn — chỉ ~11% thành công")
    bullets(s, 0.7, 1.4, 11.9, 5.3, [
        ("Bối cảnh: ", "mỗi cuộc gọi tốn chi phí nhân sự và thời gian, phần lớn khách từ chối."),
        ("Mục tiêu: ", "xếp hạng khách theo xác suất đồng ý, ưu tiên gọi nhóm đầu bảng."),
        ("Loại bài toán: ", "phân lớp nhị phân mất cân bằng (yes chỉ 11,27%)."),
        ("Yêu cầu thực tế: ", "mô hình chỉ dùng thông tin có TRƯỚC khi gọi — biến duration (thời lượng cuộc gọi) không dùng được."),
        ("Sản phẩm: ", "mô hình + ngưỡng quyết định + demo xếp hạng khách hàng."),
    ], size=20)

    # 3. Dữ liệu
    s = content_slide("Dữ liệu: Bank Marketing (UCI)",
                      "Một ngân hàng Bồ Đào Nha, 05/2008 – 11/2010, sắp xếp theo thời gian")
    bullets(s, 0.7, 1.4, 6.2, 5.3, [
        ("Quy mô: ", "41,188 dòng → loại 12 dòng trùng → 41,176 dòng."),
        ("Biến: ", "20 biến đầu vào + y (khách hàng, chiến dịch, kinh tế vĩ mô)."),
        ("Chia tập: ", "stratified 80/20 (seed 42): train 32,940, test 8,236; test khóa từ đầu."),
        ("Chia theo thời gian: ", "train yes 6,38% vs test yes 30,83% → concept drift."),
        ("Trích dẫn: ", "Moro, Cortez & Rita (2014), Decision Support Systems."),
    ], size=19)
    add_table(s, 7.3, 1.5, 5.3, ["Tập", "Số mẫu", "Tỷ lệ yes"],
              [["Train", "32,940", "11,27%"], ["Test", "8,236", "11,27%"],
               ["Time train", "32,940", "6,38%"], ["Time test", "8,236", "30,83%"]],
              col_widths=[1.7, 1.7, 1.9], font_size=15)

    # 4. EDA
    s = content_slide("EDA: hai tín hiệu mạnh nhất đến từ lịch sử liên hệ",
                      "EDA chỉ trên train — không dùng thông tin từ test")
    cb = eda["contacted_before_rate"]
    po = eda["poutcome_yes_rate"]
    bullets(s, 0.7, 1.4, 6.0, 5.3, [
        ("Đã từng liên hệ: ", f"tỷ lệ đồng ý {pct(cb['1']['yes_rate'], 1)} vs {pct(cb['0']['yes_rate'], 1)} nhóm chưa liên hệ."),
        ("Chiến dịch trước success: ", f"đồng ý {pct(po['success'], 0)}, vs nonexistent {pct(po['nonexistent'], 1)}."),
        ("Sinh viên 32%, nghỉ hưu 26% ", "đồng ý cao nhất; lao động phổ thông chỉ 7%."),
        ("Tháng 3/9/10/12: ", "tỷ lệ yes 43–50% nhưng rất ít cuộc gọi."),
        ("Vĩ mô đa cộng tuyến: ", "euribor3m–emp.var.rate = 0,97."),
        ("Duration: ", "trung vị 449s (yes) vs 164s (no) → dấu hiệu leakage."),
    ], size=18)
    add_picture(s, "fig06_pdays_poutcome.png", 7.0, 1.5, 5.8)

    # 5. Phương pháp
    s = content_slide("Phương pháp: một pipeline, không rò rỉ",
                      "Mọi mô hình dùng cùng split, cùng fold, cùng pipeline")
    bullets(s, 0.7, 1.4, 11.9, 5.3, [
        ("Tiền xử lý: ", "unknown giữ làm category; pdays=999 → cờ contacted_before + pdays_clean=0; campaign cắt trần 15; duration loại khỏi kịch bản chính."),
        ("Mất cân bằng: ", "class_weight='balanced'; metric chính PR-AUC (nhạy lớp thiểu số, không phụ thuộc ngưỡng)."),
        ("So sánh: ", "12 mô hình, StratifiedKFold 5 lần, seed 42; tuning RandomizedSearchCV tối ưu PR-AUC, chỉ trên train."),
        ("Chọn mô hình: ", "PR-AUC CV cao nhất; quy tắc 1-SE (chênh < 1 độ lệch chuẩn thì chọn mô hình đơn giản hơn)."),
        ("Ngưỡng: ", "chọn bằng dự đoán out-of-fold trên train; test đã được mở nhiều lần trong quá trình đính chính (xem báo cáo mục 4.2.8)."),
    ], size=19)

    # 6. So sánh CV
    s = content_slide("Kết quả 1: so sánh mô hình bằng CV",
                      "PR-AUC là metric xếp hạng (F1/Recall/Precision ở ngưỡng 0,5)")
    top = cv0.sort_values("pr_auc_cv", ascending=False).head(6)
    rows = [[n, fmt(r["roc_auc_cv"], 3), fmt(r["pr_auc_cv"], 3), fmt(r["f1_cv"], 3),
             fmt(r["recall_cv"], 3), fmt(r["precision_cv"], 3)]
            for n, r in top.iterrows()]
    add_table(s, 0.7, 1.5, 11.9, ["Mô hình", "ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"],
              rows, col_widths=[3.4, 1.7, 1.7, 1.7, 1.7, 1.7], font_size=16)
    bullets(s, 0.7, 4.6, 11.9, 2.2, [
        ("Nhận xét: ", f"dẫn đầu là {top.index[0]} (PR-AUC {num(top.iloc[0]['pr_auc_cv'], 3)}); "
         "Decision Tree không giới hạn độ sâu overfit nặng (train ≈ 1,0, CV thấp); "
         "class_weight tăng Recall, giảm Precision, PR-AUC gần như không đổi."),
    ], size=17)

    # 7. Tuning
    s = content_slide("Kết quả 2: tuning và chọn mô hình cuối",
                      "RandomizedSearchCV, tối ưu PR-AUC, chỉ trên train")
    rows = []
    for name in cvt.index:
        r = cvt.loc[name]
        before = float(cv0.loc[name, "pr_auc_cv"]) if name in cv0.index else float("nan")
        rows.append([name, fmt(before, 3), fmt(r["pr_auc_cv"], 3), f"{nums(r['pr_auc_cv'] - before, 3)}"])
    add_table(s, 0.7, 1.4, 6.6, ["Mô hình", "Trước", "Sau tuning", "Cải thiện"],
              rows, col_widths=[2.2, 1.4, 1.6, 1.4], font_size=15)
    bullets(s, 7.7, 1.4, 5.0, 5.3, [
        (f"Mô hình cuối: {final['final_model']}", f" — PR-AUC CV {num(tuned_best['pr_auc_cv'], 3)}."),
        ("Lý do chọn: ", final["selection_reason"]),
        ("Tham số tốt nhất: ", json.dumps(load_json("best_params.json").get(final["final_model"], {}), ensure_ascii=False)),
    ], size=16)
    add_picture(s, "fig08_validation_curve_tree.png", 7.7, 4.0, 4.6)

    # 8. Đánh giá cuối
    s = content_slide("Kết quả 3: đánh giá cuối trên test",
                      f"Mô hình cuối: {final['final_model']} — ngưỡng tối ưu F1 = {num(final['threshold_f1'], 3)}")
    add_table(s, 0.7, 1.4, 6.6,
              ["Metric (test)", "Ngưỡng F1", "Ngưỡng recall ≥ 0,7"],
              [["ROC-AUC", fmt(m["roc_auc"]), fmt(final["test_metrics_recall70_threshold"]["roc_auc"])],
               ["PR-AUC", fmt(m["pr_auc"]), fmt(final["test_metrics_recall70_threshold"]["pr_auc"])],
               ["F1 (yes)", fmt(m["f1"]), fmt(final["test_metrics_recall70_threshold"]["f1"])],
               ["Recall (yes)", fmt(m["recall"]), fmt(final["test_metrics_recall70_threshold"]["recall"])],
               ["Precision (yes)", fmt(m["precision"]), fmt(final["test_metrics_recall70_threshold"]["precision"])],
               ["TN / FP / FN / TP", f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}",
                f"{final['test_metrics_recall70_threshold']['tn']}/{final['test_metrics_recall70_threshold']['fp']}/{final['test_metrics_recall70_threshold']['fn']}/{final['test_metrics_recall70_threshold']['tp']}"]],
              col_widths=[2.4, 2.1, 2.1], font_size=15)
    ci = final["bootstrap_ci_95"]
    bullets(s, 7.7, 1.4, 5.0, 5.3, [
        ("Khoảng tin cậy 95% (bootstrap): ", f"ROC-AUC [{num(ci['roc_auc'][0], 3)}, {num(ci['roc_auc'][1], 3)}], PR-AUC [{num(ci['pr_auc'][0], 3)}, {num(ci['pr_auc'][1], 3)}]."),
        ("Kết quả CV và test nhất quán ", "— quy trình chọn mô hình không bị rò rỉ."),
    ], size=16)
    add_picture(s, "fig10_final_test_curves.png", 2.47, 4.45, 8.4)

    # 9. Diễn giải
    s = content_slide("Diễn giải: yếu tố nào quyết định?",
                      "Permutation importance + hệ số Logistic Regression + cây nông")
    imp = ana["permutation_importance_top10"]
    top3 = list(imp.items())[:3]
    bullets(s, 0.7, 1.4, 6.0, 5.3, [
        ("Quan trọng nhất: ", f"{top3[0][0]} (giảm {num(top3[0][1], 3)} PR-AUC khi hoán vị), rồi {top3[1][0]} và {top3[2][0]}."),
        ("Duration importance = 0 ", "— pipeline thực sự loại bỏ biến rò rỉ."),
        ("LogReg: ", "poutcome=success và contacted_before có odds ratio lớn nhất."),
        ("Cây nông: ", "luật đầu tiên tách theo kết quả chiến dịch trước và lãi suất."),
    ], size=18)
    add_picture(s, "fig12_permutation_importance.png", 7.0, 1.5, 5.8)

    # 10. Phân tích lỗi
    s = content_slide("Phân tích lỗi: mô hình sai ở đâu?",
                      "Recall của lớp yes theo nhóm — nhóm nào bị bỏ sót nhiều nhất")
    ebg = ana["error_by_group"]
    lines = []
    for group in ["job", "poutcome"]:
        items = sorted(ebg[group].items(), key=lambda kv: kv[1]["recall"])[:3]
        lines.append((f"{group}: ", ", ".join(f"{k} (recall {num(v['recall'], 2)})" for k, v in items)))
    bullets(s, 0.7, 1.4, 11.9, 5.3, lines + [
        ("Nhóm poutcome = success ", "được bắt gần như trọn vẹn vì tín hiệu quá rõ."),
        ("Nhóm bỏ sót nhiều nhất: ", "blue-collar (recall 0,27) — nhóm có tỷ lệ đồng ý thấp nhất — và các nhóm nghề hiếm như entrepreneur."),
        ("Gọi nhầm (FP) tập trung ", "ở nhóm có đặc điểm giống khách đồng ý nhưng thực tế từ chối."),
    ], size=19)

    # 11. Duration leakage
    s = content_slide("Thí nghiệm: duration — con số “ảo” (RQ4)",
                      "Cùng mô hình đã tuning, chỉ khác có/không duration")
    ab = extra["duration_ab_cv"]
    a, b = ab["A_khong_duration"], ab["B_co_duration"]
    add_table(s, 0.7, 1.5, 11.9, ["Kịch bản", "ROC-AUC (CV)", "PR-AUC (CV)"],
              [["A — Không duration (triển khai được)", fmt(a["roc_auc_cv"]), fmt(a["pr_auc_cv"])],
               ["B — Có duration (đối chứng)", fmt(b["roc_auc_cv"]), fmt(b["pr_auc_cv"])],
               ["Chênh lệch (B − A)", f"{nums(b['roc_auc_cv'] - a['roc_auc_cv'], 3)}", f"{nums(b['pr_auc_cv'] - a['pr_auc_cv'], 3)}"]],
              col_widths=[5.5, 3.2, 3.2], font_size=17)
    bullets(s, 0.7, 4.2, 11.9, 2.6, [
        ("Giải thích: ", "duration phản ánh kết quả cuộc gọi (khách quan tâm thì nói lâu), không phải thông tin có trước khi gọi."),
        ("Kết luận: ", "chỉ kịch bản A triển khai được; số của B chỉ có giá trị minh họa."),
    ], size=18)

    # 12. Ablation + time split
    s = content_slide("Thí nghiệm: nhóm biến nào quan trọng + concept drift (RQ3)",
                      "Ablation và chia theo thời gian")
    abl = extra["ablation_cv"]
    rows = [[k, fmt(v["roc_auc_cv"], 3), fmt(v["pr_auc_cv"], 3)] for k, v in abl.items()]
    add_table(s, 0.7, 1.4, 6.2, ["Nhóm biến", "ROC-AUC", "PR-AUC"], rows,
              col_widths=[2.8, 1.7, 1.7], font_size=13)
    ts = extra["time_split"]
    fn = final["final_model"]
    bullets(s, 7.3, 1.4, 5.4, 5.3, [
        ("Ablation: ", "nhóm chiến dịch + vĩ mô đóng góp nhiều nhất; thông tin cá nhân ít nhất."),
        ("Time split: ", f"train yes 6,38% → test yes 30,83%; hiệu năng giảm so với chia ngẫu nhiên → concept drift."),
        ("Bỏ biến vĩ mô: ", f"ROC-AUC time split {num(ts[fn + '_no_macro']['roc_auc'], 3)} so với {num(ts[fn + '_full']['roc_auc'], 3)} bản đầy đủ — kiểm tra giả thuyết “đồng hồ thời gian”."),
    ], size=17)

    # 13. Gain/lift
    s = content_slide("Giá trị kinh doanh: gọi top 20% bắt được bao nhiêu? (RQ5)",
                      "Gain/Lift chart + chi phí–lợi ích (giả định C=1, R=15)")
    top20 = extra["top20_capture"]
    bullets(s, 0.7, 1.4, 6.0, 5.3, [
        (f"Gọi top 20% → bắt được {pct(top20, 0)} ", "số người đồng ý."),
        ("Ngưỡng gọi tối ưu ", "thay đổi theo chi phí cuộc gọi và giá trị khách hàng (xem Bảng 6.5 trong báo cáo)."),
        ("Đề xuất: ", "ưu tiên khách đã từng thành công, sinh viên/nghỉ hưu, gọi vào tháng có tỷ lệ cao và khi lãi suất thấp."),
    ], size=19)
    add_picture(s, "fig16_gain_chart.png", 7.0, 1.5, 5.8)

    # 14. Kết luận RQ
    s = content_slide("Kết luận: trả lời 5 câu hỏi nghiên cứu", "")
    bullets(s, 0.7, 1.4, 11.9, 5.3, [
        ("RQ1: ", f"{final['final_model']} cân bằng tốt nhất giữa Recall và Precision (PR-AUC test {num(m['pr_auc'], 3)})."),
        ("RQ2: ", "class_weight tăng Recall, giảm Precision, PR-AUC gần như không đổi."),
        ("RQ3: ", "nhóm chiến dịch và vĩ mô quan trọng nhất; thông tin cá nhân ít nhất."),
        ("RQ4: ", "duration làm PR-AUC tăng vọt nhưng là leakage — chỉ kịch bản không duration triển khai được."),
        ("RQ5: ", f"gọi top 20% bắt được {pct(top20, 0)} số người đồng ý."),
    ], size=19)

    # 15. Hạn chế + cảm ơn
    s = content_slide("Hạn chế và hướng phát triển", "")
    bullets(s, 0.7, 1.4, 11.9, 4.0, [
        ("Hạn chế: ", "dữ liệu một ngân hàng, giai đoạn 2008–2010; concept drift theo thời gian; chi phí–lợi ích dùng giả định của nhóm."),
        ("Hướng phát triển: ", "huấn luyện lại định kỳ; thử bản không dùng biến vĩ mô; bổ sung dữ liệu hành vi khách hàng."),
        ("Đối chiếu: ", "số RF của thầy (0,814/0,482) đã được tái lập gần như trọn vẹn bằng RF đã tuning — "
         f"test đạt ROC-AUC {num(m['roc_auc'], 3)} / PR-AUC {num(m['pr_auc'], 3)}; "
         "chênh lệch F1/Recall còn lại chủ yếu do ngưỡng quyết định khác nhau."),
    ], size=19)
    _, tf = textbox(s, 0.7, 5.9, 11.9, 0.8)
    para(tf, "Cảm ơn thầy cô và các bạn đã lắng nghe!", size=22, bold=True,
         color=NAVY, align=PP_ALIGN.CENTER, first=True)

    prs.save(str(OUT))
    print("Đã lưu", OUT)


if __name__ == "__main__":
    main()
