# -*- coding: utf-8 -*-
"""
Tự động tạo BÁO CÁO CUỐI KỲ (.docx) — Nhóm 14, môn Máy học
Đề tài: Dự đoán khách hàng đăng ký tiền gửi có kỳ hạn (Bank Marketing)

Chạy:  python scripts/build_report.py   (từ thư mục gốc dự án)
Yêu cầu: pip install python-docx
Đầu ra: reports/BaoCao_CuoiKy_Nhom14_BankMarketing.docx

Mọi số liệu trong báo cáo đều lấy từ các file kết quả do các script
giai đoạn sinh ra (reports/*.json, reports/*.csv) — KHÔNG gõ tay số.
Định dạng: A4, Times New Roman 13, giãn dòng 1.5, tiêu đề đen đậm.
"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
OUT = REPORTS / "BaoCao_CuoiKy_Nhom14_BankMarketing.docx"

FONT = "Times New Roman"
BODY_SIZE = 13
HEADER_FILL = "D9E2F3"

doc = Document()


def load_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%".replace(".", ",")


def num(x, nd=4):
    """Định dạng số với dấu phẩy thập phân (quy ước tiếng Việt)."""
    return f"{x:.{nd}f}".replace(".", ",")


def nums(x, nd=4):
    """Như num() nhưng giữ dấu + / - (dùng cho chênh lệch)."""
    return f"{x:+.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ helpers
def setup_page():
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.0)
    sec.header_distance, sec.footer_distance = Cm(1.25), Cm(1.25)


def set_base_style():
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(BODY_SIZE)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    for name, size, color, caps in [
        ("Heading 1", 14, RGBColor(0, 0, 0), True),
        ("Heading 2", 13, RGBColor(0, 0, 0), False),
        ("Heading 3", 13, RGBColor(0, 0, 0), False),
    ]:
        s = doc.styles[name]
        s.font.name = FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = color
        s.font.all_caps = caps
        s.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        s.paragraph_format.space_after = Pt(6)
        s.paragraph_format.line_spacing = 1.5
        s.paragraph_format.keep_with_next = True


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def para(text="", bold=False, italic=False, size=BODY_SIZE, align=None,
         space_after=6, space_before=0, color=None, caps=False):
    p = doc.add_paragraph()
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
        r.font.name = FONT
        if color is not None:
            r.font.color.rgb = color
        if caps:
            r.font.all_caps = True
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.5
    return p


def body(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(BODY_SIZE)
    return p


def mixed(parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    for text, bold, italic in parts:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.name = FONT
        r.font.size = Pt(BODY_SIZE)
    return p


def bullet(text, bold_prefix=""):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.5
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(BODY_SIZE)
    r = p.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(BODY_SIZE)
    return p


def h1(number, title):
    return doc.add_heading(f"{number}. {title}", level=1)


def h2(number, title):
    return doc.add_heading(f"{number} {title}", level=2)


def h3(number, title):
    return doc.add_heading(f"{number} {title}", level=3)


def add_table(headers, rows, widths=None, font_size=11.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.name = FONT
        r.font.size = Pt(font_size)
        shade(hdr[i], HEADER_FILL)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if isinstance(val, (int, float))
                           else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(str(val))
            r.font.name = FONT
            r.font.size = Pt(font_size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.italic = True
    r.font.name = FONT
    r.font.size = Pt(11.5)
    return p


def figure(filename, cap, width_cm=16.0):
    path = FIG / filename
    if not path.exists():
        para(f"[Thiếu hình: {filename}]", italic=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(cap)


def add_toc():
    p = doc.add_paragraph()
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "separate")
    fld3 = OxmlElement("w:fldChar"); fld3.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2); run._r.append(fld3)
    para("Nhấn chuột phải vào mục lục rồi chọn Update Field để hiện số trang.",
         italic=True, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)


def add_page_number(run):
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)


def add_footer():
    sec = doc.sections[0]
    footer = sec.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Báo cáo cuối kỳ môn Máy học — Nhóm 14 — Bank Marketing  |  Trang ")
    r.font.name = FONT
    r.font.size = Pt(11)
    add_page_number(p.add_run())


# ======================================================================
# NỘI DUNG BÁO CÁO
# ======================================================================
def cover_and_front():
    setup_page()
    set_base_style()
    add_footer()
    para("TRƯỜNG ĐẠI HỌC SÀI GÒN", bold=True, size=14,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para("KHOA CÔNG NGHỆ THÔNG TIN", bold=True, size=14,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para("BÁO CÁO CUỐI KỲ", bold=True, size=20,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para("MÔN MÁY HỌC", bold=True, size=16,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para("Đề tài:", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para("DỰ ĐOÁN KHÁCH HÀNG ĐĂNG KÝ TIỀN GỬI CÓ KỲ HẠN",
         bold=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para("(BANK MARKETING — UCI MACHINE LEARNING REPOSITORY)",
         bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
    para("Nhóm 14 — Lớp DCT124C3", bold=True, size=14,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para("Thành viên thực hiện:", bold=True, size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para("Trần Quang Minh — MSSV 3124411168", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para("________________________________________ — MSSV ____________",
         size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para("________________________________________ — MSSV ____________",
         size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para("Giảng viên hướng dẫn: ________________________________________",
         size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
    para("TP. Hồ Chí Minh, tháng 11 năm 2026", italic=True, size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()
    h1("", "MỤC LỤC")
    add_toc()
    doc.add_page_break()


def chapter_1(eda):
    h1("1", "MỞ ĐẦU")
    h2("1.1", "Bối cảnh và động lực")
    body("Ngân hàng triển khai chiến dịch telemarketing để mời khách hàng gửi tiền "
         "có kỳ hạn. Mỗi cuộc gọi tốn chi phí nhân sự và thời gian, trong khi chỉ "
         f"khoảng {pct(eda['train_yes_rate'])} số cuộc gọi thành công. Nếu ngân hàng "
         "gọi ngẫu nhiên, phần lớn chi phí rơi vào những khách hàng gần như chắc chắn "
         "từ chối. Ngược lại, nếu xếp hạng được khách hàng theo khả năng đồng ý, ngân "
         "hàng có thể tập trung nguồn lực vào nhóm đầu bảng, giảm chi phí mà vẫn giữ "
         "được phần lớn doanh thu. Đó chính là bài toán của đồ án này.")
    body("Bộ dữ liệu Bank Marketing của UCI ghi lại kết quả các chiến dịch gọi điện "
         "của một ngân hàng Bồ Đào Nha trong giai đoạn 05/2008–11/2010, với thông tin "
         "khách hàng, thông tin chiến dịch và các chỉ số kinh tế vĩ mô tại thời điểm "
         "gọi. Nhiệm vụ của nhóm là xây dựng mô hình phân lớp nhị phân dự đoán khách "
         "hàng có đăng ký tiền gửi kỳ hạn hay không, đồng thời trả lời năm câu hỏi "
         "nghiên cứu gắn với các thí nghiệm cụ thể.")
    h2("1.2", "Mục tiêu và câu hỏi nghiên cứu")
    body("Mục tiêu tổng quát: xây dựng mô hình dự đoán khách hàng đăng ký tiền gửi "
         "kỳ hạn có thể dùng được trong thực tế (tức là chỉ dùng thông tin có trước "
         "khi gọi), đánh giá một cách trung thực bằng tập test được khóa từ đầu, và "
         "chuyển kết quả thành khuyến nghị kinh doanh cụ thể.")
    add_table(
        ["Mã", "Câu hỏi nghiên cứu", "Thí nghiệm trả lời"],
        [["RQ1", "Mô hình nào cân bằng tốt nhất giữa Recall và Precision của lớp yes trong điều kiện mất cân bằng?",
          "So sánh các mô hình bằng cross-validation, chốt trên test"],
         ["RQ2", "Xử lý mất cân bằng (class_weight) cải thiện Recall/F1 bao nhiêu và đánh đổi gì về Precision?",
          "Cùng mô hình, có và không có class_weight"],
         ["RQ3", "Nhóm yếu tố nào ảnh hưởng nhất: thông tin khách, lịch sử chiến dịch hay kinh tế vĩ mô?",
          "Ablation theo nhóm biến + permutation importance + hệ số LogReg"],
         ["RQ4", "Biến duration ảnh hưởng thế nào đến hiệu năng và vì sao không dùng được trong thực tế?",
          "Kịch bản A (không duration) so với B (có duration)"],
         ["RQ5", "Nếu chỉ gọi top 20% khách theo điểm mô hình thì bắt được bao nhiêu % người đồng ý?",
          "Gain/Lift chart + tính chi phí–lợi ích"]],
        widths=[1.2, 7.0, 6.8], font_size=11)
    para("Bảng 1.1. Câu hỏi nghiên cứu và thí nghiệm trả lời.", italic=True, size=11.5,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    h2("1.3", "Đóng góp chính của đồ án")
    bullet("Xây dựng pipeline tiền xử lý duy nhất, chống rò rỉ dữ liệu, dùng chung "
           "cho mọi mô hình và dùng thẳng được trong demo.", bold_prefix="Quy trình chuẩn: ")
    bullet("So sánh công bằng 12 mô hình bằng cùng split, cùng fold, cùng pipeline; "
           "tuning bằng RandomizedSearchCV tối ưu PR-AUC chỉ trên train.",
           bold_prefix="So sánh và tuning: ")
    bullet("Chọn ngưỡng quyết định bằng dự đoán out-of-fold trên train, đánh giá cuối "
           "trên test, kèm khoảng tin cậy bootstrap. (Đính chính 19/11/2026: test đã được mở nhiều lần trong toàn bộ dự án — xem báo cáo cuối kỳ mục 4.2.8.)",
           bold_prefix="Đánh giá trung thực: ")
    bullet("Thí nghiệm bổ sung: bằng chứng định lượng về target leakage của duration, "
           "ablation theo nhóm biến, kiểm tra concept drift bằng chia theo thời gian "
           "(kèm giả thuyết biến vĩ mô là “đồng hồ thời gian”), hiệu chỉnh xác suất và "
           "phân tích gain/lift kèm chi phí–lợi ích.", bold_prefix="Thí nghiệm bổ sung: ")
    h2("1.4", "Cấu trúc báo cáo")
    body("Chương 2 trình bày cơ sở lý thuyết về các mô hình sử dụng. Chương 3 mô tả "
         "dữ liệu và kết quả khám phá dữ liệu (EDA). Chương 4 trình bày phương pháp: "
         "pipeline tiền xử lý, thiết kế thực nghiệm, metric và quy tắc chọn mô hình. "
         "Chương 5 báo cáo kết quả so sánh, tuning, chọn ngưỡng và đánh giá cuối. "
         "Chương 6 phân tích lỗi, diễn giải mô hình và các thí nghiệm bổ sung. "
         "Chương 7 kết luận và trả lời các câu hỏi nghiên cứu.")


def chapter_2():
    h1("2", "CƠ SỞ LÝ THUYẾT")
    h2("2.1", "Phân lớp nhị phân mất cân bằng")
    body("Bài toán là phân lớp nhị phân với lớp thiểu số (yes) chỉ chiếm khoảng "
         "11%. Trong điều kiện này, accuracy gây hiểu lầm: một mô hình đoán toàn "
         "“no” đã đạt accuracy khoảng 89% mà không bắt được khách hàng tiềm năng nào. "
         "Vì vậy đồ án dùng PR-AUC (diện tích dưới đường Precision–Recall) làm metric "
         "chính để xếp hạng và tuning — metric này nhạy với lớp thiểu số và không phụ "
         "thuộc ngưỡng — cùng ROC-AUC, Recall, Precision và F1 của lớp yes để đối "
         "chiếu. Các metric phụ thuộc ngưỡng (Recall, Precision, F1) chỉ được so sánh "
         "sau khi mọi mô hình đã qua cùng một cơ chế chọn ngưỡng (xem Chương 4).")
    h2("2.2", "Các mô hình sử dụng")
    add_table(
        ["Mô hình", "Nguyên lý ngắn", "Ưu điểm", "Hạn chế", "Lý do chọn"],
        [["Logistic Regression", "Đường biên tuyến tính trên log-odds, tối ưu log-loss",
          "Nhanh, hệ số dễ diễn giải, cho xác suất", "Chỉ biên tuyến tính, nhạy đa cộng tuyến",
          "Baseline mạnh và diễn giải"],
         ["Decision Tree", "Chia dữ liệu theo luật if-then để giảm tạp chất (gini)",
          "Trực quan, phi tuyến, không cần scale", "Dễ overfit, phương sai cao",
          "Minh họa overfitting và luật kinh doanh"],
         ["Random Forest", "Tập hợp nhiều cây (bagging + chọn ngẫu nhiên biến)",
          "Giảm phương sai, bền với nhiễu", "Khó diễn giải hơn, chậm hơn",
          "Cải thiện cây đơn, có feature importance"],
         ["KNN", "Dự đoán theo k láng giềng gần nhất",
          "Đơn giản, không giả định phân phối", "Chậm khi dự đoán, nhạy thang đo và chiều cao (one-hot)",
          "Đại diện nhóm dựa trên khoảng cách"],
         ["LinearSVC", "Tìm siêu phẳng lề lớn nhất",
          "Hiệu quả với chiều cao", "Không cho xác suất trực tiếp",
          "Đại diện nhóm biên lề"],
         ["HistGradientBoosting", "Gradient boosting trên histogram",
          "Mạnh trên dữ liệu bảng, nhanh", "Khó diễn giải, nhiều siêu tham số",
          "Mô hình tham chiếu mạnh"],
         ["XGBoost / LightGBM", "Gradient boosting (tối ưu cây, histogram)",
          "Chuẩn mạnh nhất trên dữ liệu bảng", "Cần cài thêm thư viện",
          "Đối chiếu với HGB"]],
        widths=[2.6, 3.6, 2.8, 3.0, 3.0], font_size=10.5)
    para("Bảng 2.1. Các mô hình sử dụng và lý do chọn.", italic=True, size=11.5,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    body("Nhóm không chạy mô hình tràn lan: mỗi mô hình có lý do chọn như bảng trên, "
         "theo nguyên tắc 6 của đề. DummyClassifier (strategy='prior') làm mốc tối "
         "thiểu: ROC-AUC bằng 0,5 và PR-AUC bằng tỷ lệ yes.")
    h2("2.3", "Xử lý mất cân bằng lớp")
    body("Đồ án dùng class_weight='balanced' (tăng trọng số lớp thiểu số theo tỷ lệ "
         "nghịch với tần suất) thay vì lấy mẫu lại dữ liệu, vì cách này đơn giản, "
         "không làm đổi dữ liệu và so sánh công bằng giữa các mô hình. Tác dụng của "
         "class_weight chủ yếu là dịch ngưỡng quyết định ngầm: Recall tăng mạnh, "
         "Precision giảm, còn PR-AUC và ROC-AUC (không phụ thuộc ngưỡng) gần như không "
         "đổi — hiện tượng này được kiểm chứng ở Chương 5. Thí nghiệm SMOTE được làm "
         "thêm ở Chương 6 để đối chiếu (chỉ lấy mẫu lại trong fold train).")
    h2("2.4", "Target leakage và biến duration")
    body("Biến duration (thời lượng cuộc gọi, tính bằng giây) chỉ biết được sau khi "
         "cuộc gọi kết thúc. Cuộc gọi dài thường là khách đang quan tâm, nên duration "
         "tương quan rất mạnh với kết quả — nhưng nó là hệ quả của sự quan tâm chứ "
         "không phải thông tin có trước khi gọi. Dùng duration để quyết định “ai nên "
         "được gọi” là target leakage: mô hình trông rất tốt trên giấy nhưng không "
         "thể triển khai. Chính tác giả bộ dữ liệu cũng khuyến cáo bỏ biến này khi "
         "muốn mô hình thực tế. Đồ án thiết kế hai kịch bản: A (chính, không duration) "
         "và B (đối chứng, có duration) để đo lường định lượng mức độ rò rỉ.")


def chapter_3(eda):
    h1("3", "DỮ LIỆU VÀ KHÁM PHÁ DỮ LIỆU")
    h2("3.1", "Nguồn dữ liệu và mô tả chung")
    body("Dữ liệu lấy từ UCI Machine Learning Repository, bộ Bank Marketing "
         "(https://archive.ics.uci.edu/dataset/222/bank+marketing, "
         "DOI 10.24432/C5K306), file "
         "bank-additional-full.csv. Bộ dữ liệu ghi lại các chiến dịch telemarketing "
         "của một ngân hàng Bồ Đào Nha trong giai đoạn 05/2008–11/2010, sắp xếp theo "
         "thời gian. Sau khi loại 12 dòng trùng, còn 41,176 dòng với 20 biến đầu vào "
         "và biến mục tiêu y (yes/no đăng ký tiền gửi kỳ hạn). Trích dẫn bắt buộc: "
         "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict "
         "the success of bank telemarketing. Decision Support Systems, 62, 22–31.")
    add_table(
        ["Nhóm", "Biến", "Kiểu", "Ý nghĩa và lưu ý"],
        [["Khách hàng", "age", "Số", "Tuổi"],
         ["", "job", "Phân loại", "Nghề nghiệp (admin., blue-collar, student, retired...)"],
         ["", "marital", "Phân loại", "Tình trạng hôn nhân"],
         ["", "education", "Phân loại", "Trình độ học vấn"],
         ["", "default", "Phân loại", "Có nợ xấu không (rất ít yes); unknown 20,82%"],
         ["", "housing", "Phân loại", "Có vay mua nhà không"],
         ["", "loan", "Phân loại", "Có vay cá nhân không"],
         ["Chiến dịch", "contact", "Phân loại", "Kênh: cellular hoặc telephone"],
         ["", "month", "Phân loại", "Tháng liên hệ cuối"],
         ["", "day_of_week", "Phân loại", "Thứ liên hệ cuối"],
         ["", "duration", "Số", "Thời lượng cuộc gọi (giây). Chỉ biết sau khi gọi — gây leakage"],
         ["", "campaign", "Số", "Số lần liên hệ trong chiến dịch này (lệch phải, có outlier)"],
         ["", "pdays", "Số", "Số ngày từ lần liên hệ chiến dịch trước. 999 = chưa từng liên hệ"],
         ["", "previous", "Số", "Số lần liên hệ trước chiến dịch này"],
         ["", "poutcome", "Phân loại", "Kết quả chiến dịch trước (failure, nonexistent, success)"],
         ["Vĩ mô", "emp.var.rate", "Số", "Tỷ lệ biến động việc làm (theo quý)"],
         ["", "cons.price.idx", "Số", "Chỉ số giá tiêu dùng (theo tháng)"],
         ["", "cons.conf.idx", "Số", "Chỉ số niềm tin người tiêu dùng (theo tháng)"],
         ["", "euribor3m", "Số", "Lãi suất euribor 3 tháng (theo ngày)"],
         ["", "nr.employed", "Số", "Số lao động (theo quý)"],
         ["Mục tiêu", "y", "Nhị phân", "yes/no đăng ký tiền gửi kỳ hạn"]],
        widths=[2.0, 2.6, 1.8, 8.6], font_size=10.5)
    para("Bảng 3.1. Mô tả các biến trong bộ dữ liệu Bank Marketing.", italic=True,
         size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    h2("3.2", "Chia dữ liệu và khóa tập test")
    body(f"Tập train có {eda['train_rows']:,} mẫu, tập test có {eda['test_rows']:,} mẫu "
         f"(chia stratified 80/20, random_state=42), tỷ lệ yes đều "
         f"{pct(eda['train_yes_rate'])}. Tập test được khóa ngay từ đầu và chỉ dùng "
         "ở bước đánh giá cuối. EDA trong chương này thực hiện CHỈ trên tập train, "
         "theo khuyến nghị của Cẩm nang, để không dùng thông tin từ test cho bất kỳ "
         "quyết định nào. Ngoài ra, dữ liệu còn được chia theo thời gian (80% đầu làm "
         "train, 20% cuối làm test) để kiểm tra độ ổn định: ở cách chia này, tỷ lệ yes "
         "là 6,38% ở train và 30,83% ở test — dấu hiệu của sự dịch chuyển phân phối "
         "theo thời gian, được phân tích kỹ ở Chương 6.")
    h2("3.3", "Kết quả khám phá dữ liệu (EDA)")
    h3("3.3.1", "Mất cân bằng lớp và phân phối biến số")
    body(f"Biến mục tiêu mất cân bằng rõ rệt: chỉ {pct(eda['train_yes_rate'])} khách "
         "đăng ký. Vì vậy accuracy không dùng được làm metric chính (đoán toàn “no” "
         "đã được khoảng 89%). Các biến campaign và previous lệch phải mạnh "
         f"(campaign có độ lệch {num(eda['numeric_describe']['campaign']['skew'], 3)}, phân vị "
         f"99 là {num(eda['numeric_describe']['campaign']['p99'], 1)}); chỉ "
         f"{pct(eda['campaign_gt15_rate'])} số khách bị gọi trên 15 lần, và nhóm này "
         f"hầu như không đồng ý (tỷ lệ yes {pct(eda['campaign_gt15_yes_rate'])}). "
         "Điều này dẫn đến quyết định cắt trần campaign ở 15.")
    figure("fig01_target_distribution.png",
           "Hình 3.1. Phân phối biến mục tiêu y trên tập train.")
    figure("fig02_numeric_histograms.png",
           "Hình 3.2. Phân phối các biến số trên tập train.")
    h3("3.3.2", "Nhóm khách nào dễ đồng ý hơn?")
    cb = eda["contacted_before_rate"]
    po = eda["poutcome_yes_rate"]
    body(f"Hai tín hiệu mạnh nhất đến từ lịch sử liên hệ. Khách đã từng được liên hệ "
         f"trước đó (pdays khác 999, chỉ {cb['1']['n']:,} người) có tỷ lệ đồng ý "
         f"{pct(cb['1']['yes_rate'])}, gấp khoảng "
         f"{num(cb['1']['yes_rate'] / cb['0']['yes_rate'], 1)} lần nhóm chưa từng liên hệ "
         f"({pct(cb['0']['yes_rate'])}). Khách có kết quả chiến dịch trước là success "
         f"đồng ý tới {pct(po['success'])}, trong khi nhóm nonexistent chỉ "
         f"{pct(po['nonexistent'])} và failure {pct(po['failure'])}. Về nghề nghiệp, "
         "sinh viên (32,3%) và người nghỉ hưu (25,8%) đồng ý cao hơn hẳn mặt bằng, "
         "trong khi lao động phổ thông (blue-collar) chỉ 6,9%. Về thời điểm, các tháng "
         "3, 9, 10, 12 có tỷ lệ yes 43–50% nhưng số cuộc gọi rất ít (ví dụ tháng 3 "
         "chỉ 546 cuộc trong toàn bộ dữ liệu), còn tháng 5 — tháng gọi nhiều nhất — "
         "chỉ đạt 6,4%.")
    figure("fig04_yes_rate_by_category.png",
           "Hình 3.3. Tỷ lệ đăng ký theo các biến phân loại trên tập train.")
    figure("fig06_pdays_poutcome.png",
           "Hình 3.4. Tỷ lệ đăng ký theo contacted_before và poutcome.")
    h3("3.3.3", "Biến vĩ mô đa cộng tuyến và biến duration")
    mc = eda["macro_correlations"]
    body(f"Ba biến vĩ mô tương quan rất mạnh với nhau: euribor3m–emp.var.rate "
         f"{num(mc['euribor3m-emp.var.rate'], 3)}, euribor3m–nr.employed "
         f"{num(mc['euribor3m-nr.employed'], 3)}, emp.var.rate–nr.employed "
         f"{num(mc['emp.var.rate-nr.employed'], 3)}. Với mô hình tuyến tính, đa cộng tuyến "
         "làm hệ số của từng biến riêng lẻ không đáng tin; với mô hình cây thì không "
         "ảnh hưởng. Nhóm quyết định giữ cả ba biến cho mô hình cây và dùng L2 cho "
         "LogReg, đồng thời cảnh báo rõ khi diễn giải hệ số.")
    dur = eda["duration_by_y"]
    body(f"Biến duration khác biệt rất lớn giữa hai lớp: trung vị {num(dur['1']['median'], 0)} "
         f"giây ở nhóm yes so với {num(dur['0']['median'], 0)} giây ở nhóm no. Đây là "
         "dấu hiệu đầu tiên của target leakage: cuộc gọi dài thường là khách đang "
         "quan tâm, nhưng thời lượng chỉ biết sau khi gọi xong nên không thể dùng để "
         "quyết định ai nên được gọi. Kịch bản B ở Chương 6 sẽ đo lường chính xác "
         "mức độ “ảo” này.")
    figure("fig05_correlation_heatmap.png",
           "Hình 3.5. Ma trận tương quan giữa các biến số trên tập train.")
    figure("fig03_boxplot_by_class.png",
           "Hình 3.6. Phân phối age, duration, campaign theo lớp y.")
    h3("3.3.4", "Giá trị unknown và bảng quyết định tiền xử lý")
    unk = eda["unknown_rate"]
    body(f"Tỷ lệ unknown cao nhất ở default ({pct(unk['default'])}), tiếp theo là "
         f"education ({pct(unk['education'])}), còn job, marital, housing, loan đều "
         "dưới 3%. Nhóm quyết định giữ unknown làm một category riêng (one-hot) thay "
         "vì loại bỏ, vì nhóm unknown có thể mang thông tin và tỷ lệ thiếu không quá "
         "lớn ở hầu hết các cột. Toàn bộ quyết định tiền xử lý rút ra từ EDA được "
         "tổng hợp trong Bảng 3.2.")
    add_table(
        ["Vấn đề phát hiện", "Quyết định", "Lý do"],
        [["Dòng trùng (12 dòng)", "Xóa trước khi chia", "Tránh trùng giữa train và test"],
         ["unknown", "Giữ làm category riêng, one-hot", "Có thể mang thông tin; tỷ lệ thiếu không quá lớn ở hầu hết cột"],
         ["pdays = 999", "Tạo cờ contacted_before, đặt pdays_clean = 0 khi chưa liên hệ", "999 làm méo thang đo của mô hình tuyến tính/khoảng cách; cờ giữ lại thông tin"],
         ["campaign đuôi dài", "Cắt trần ở 15 (giá trị cố định)", "Giảm ảnh hưởng ngoại lệ cho LogReg/KNN; cây không bị ảnh hưởng. Giá trị cố định nên không rò rỉ"],
         ["Đa cộng tuyến vĩ mô", "Giữ cho cây; với LogReg dùng L2 và phân tích hệ số cẩn thận", "Cây miễn nhiễm; L2 ổn định hệ số"],
         ["duration", "Loại khỏi kịch bản A", "Data leakage"],
         ["Mất cân bằng", "class_weight='balanced' + metric phù hợp + chọn ngưỡng", "Đơn giản, không làm đổi dữ liệu, so sánh công bằng"],
         ["Thang đo khác nhau", "StandardScaler cho biến số (trong pipeline)", "LogReg/KNN/SVM nhạy với thang đo"]],
        widths=[3.2, 5.6, 6.2], font_size=10.5)
    para("Bảng 3.2. Bảng quyết định tiền xử lý rút ra từ EDA.",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)


def chapter_4():
    h1("4", "PHƯƠNG PHÁP")
    h2("4.1", "Pipeline tiền xử lý duy nhất")
    body("Mọi mô hình dùng chung một pipeline trong src/features.py: dữ liệu thô vào, "
         "xác suất ra. Các biến đổi không học từ dữ liệu (tạo cờ contacted_before, "
         "pdays_clean = 0 khi pdays = 999, cắt trần campaign ở 15) đặt trong hàm "
         "add_features; các biến đổi học từ dữ liệu (StandardScaler cho biến số, "
         "OneHotEncoder cho biến phân loại) nằm trong ColumnTransformer và chỉ fit "
         "trên fold train. Nhờ đó, cross-validation tự động fit lại từng fold và không "
         "có rò rỉ dữ liệu. Pipeline chỉ chọn các cột trong danh sách NUM và CAT nên "
         "kịch bản A tự động không dùng duration; kịch bản B truyền thêm duration vào "
         "danh sách biến số. Cây quyết định và Random Forest không cần scale, nhưng "
         "việc scale cũng vô hại và dùng chung pipeline giúp so sánh công bằng.")
    h2("4.2", "Thiết kế thực nghiệm")
    body("Tập train (80%) dùng để so sánh mô hình, tuning và chọn ngưỡng bằng "
         "StratifiedKFold 5 lần (seed 42); tập test (20%) khóa lại trong giai đoạn giữa kỳ; "
         "ở giai đoạn cuối test đã được mở nhiều lần trong quá trình đính chính "
         "(xem báo cáo cuối kỳ mục 4.2.8). Mọi mô hình dùng đúng một đối tượng CV này để so sánh "
         "công bằng. Ngoài ra, dữ liệu còn được chia theo thời gian để kiểm tra độ ổn "
         "định (xem Chương 6).")
    add_table(
        ["Nguy cơ rò rỉ", "Cách kiểm soát"],
        [["Scaler/encoder học từ test", "Nằm trong Pipeline, chỉ fit trên fold train"],
         ["Dòng trùng ở cả train và test", "Loại trùng trước khi chia"],
         ["Biến duration", "Loại khỏi kịch bản A, chỉ dùng ở kịch bản B để đối chứng"],
         ["Chọn ngưỡng/tuning bằng test", "Chỉ dùng out-of-fold trên train"],
         ["Cắt trần campaign", "Dùng hằng số cố định (15), không tính từ test"],
         ["Lấy mẫu lại (SMOTE)", "Chỉ trong fold train, dùng imblearn Pipeline"]],
        widths=[5.0, 10.0], font_size=11)
    h2("4.3", "Metric và quy tắc chọn mô hình cuối")
    body("Metric chính là PR-AUC (average_precision) vì nhạy với lớp thiểu số và "
         "không phụ thuộc ngưỡng; ROC-AUC dùng để đối chiếu với tài liệu; Recall, "
         "Precision và F1 của lớp yes được báo cáo sau khi chọn ngưỡng. Quy tắc chọn "
         "mô hình cuối (viết ra trước khi chạy): xếp hạng theo trung bình PR-AUC trong "
         "CV sau tuning; nếu chênh lệch giữa hai mô hình nhỏ hơn một độ lệch chuẩn qua "
         "các fold thì chọn mô hình đơn giản và dễ diễn giải hơn (quy tắc 1-SE); kiểm "
         "tra thêm độ ổn định (std qua fold) và khoảng cách train–CV (overfitting).")
    h2("4.4", "Chọn ngưỡng quyết định")
    body("Ngưỡng được chọn bằng dự đoán out-of-fold trên train (cross_val_predict, "
         "5 fold): hai chính sách được tính — ngưỡng tối ưu F1 (cân bằng) và ngưỡng "
         "giữ recall tối thiểu 0,7 (ưu tiên không bỏ sót khách tiềm năng). Ngân hàng "
         "chọn chính sách theo chi phí cuộc gọi và giá trị khách hàng. Ngưỡng tối ưu "
         "F1 được dùng làm ngưỡng mặc định cho demo và đánh giá cuối.")


def chapter_5():
    import pandas as pd
    cv0 = pd.read_csv(REPORTS / "cv_default_models.csv", index_col=0)
    cvt = pd.read_csv(REPORTS / "cv_tuned_models.csv", index_col=0)
    tuned_best = load_json("tuned_best_model.json")
    final = load_json("final_summary.json")
    test_table = pd.read_csv(REPORTS / "test_results.csv", index_col=0)

    h1("5", "KẾT QUẢ THỰC NGHIỆM")
    h2("5.1", "So sánh mô hình mặc định bằng cross-validation")
    body("Bảng 5.1 so sánh 12 mô hình (kèm baseline Dummy) bằng StratifiedKFold 5 "
         "lần trên tập train, cùng pipeline tiền xử lý. Metric xếp hạng là PR-AUC; "
         "các cột _train cho thấy mức độ overfitting (điểm train cao hơn nhiều so "
         "với CV nghĩa là overfit).")
    rows = []
    order = cv0.sort_values("pr_auc_cv", ascending=False).index.tolist()
    for name in order:
        r = cv0.loc[name]
        rows.append([name, f"{num(r['roc_auc_cv'], 4)} ± {num(r['roc_auc_std'], 4)}",
                     f"{num(r['pr_auc_cv'], 4)} ± {num(r['pr_auc_std'], 4)}",
                     f"{num(r['f1_cv'], 4)}", f"{num(r['recall_cv'], 4)}",
                     f"{num(r['precision_cv'], 4)}", f"{num(r['fit_time_s'], 1)}"])
    add_table(["Mô hình", "ROC-AUC (CV)", "PR-AUC (CV)", "F1", "Recall", "Precision", "Thời gian fit (s)"],
              rows, widths=[3.0, 2.4, 2.4, 1.4, 1.4, 1.6, 2.0], font_size=10.5)
    para("Bảng 5.1. So sánh mô hình mặc định bằng 5-Fold CV trên train "
         "(F1/Recall/Precision ở ngưỡng 0,5).", italic=True, size=11.5,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    best0 = order[0]
    dummy_pr = float(cv0.loc["Dummy", "pr_auc_cv"])
    body(f"Mô hình tốt nhất ở bước mặc định là {best0} với PR-AUC CV "
         f"{num(cv0.loc[best0, 'pr_auc_cv'], 4)}, cao hơn rõ rệt so với baseline Dummy "
         f"(PR-AUC {num(dummy_pr, 4)}, bằng đúng tỷ lệ yes). Mọi mô hình đều vượt xa "
         "mốc tối thiểu này, chứng tỏ tín hiệu trong dữ liệu là thật.")
    # Overfitting: Tree train vs CV
    tree_tr = float(cv0.loc["Tree", "pr_auc_train"])
    tree_cv = float(cv0.loc["Tree", "pr_auc_cv"])
    knn_pr = float(cv0.loc["KNN", "pr_auc_cv"])
    knn_roc = float(cv0.loc["KNN", "roc_auc_cv"])
    body(f"Decision Tree không giới hạn độ sâu là ví dụ overfitting kinh điển: "
         f"PR-AUC trên train {num(tree_tr, 4)} nhưng CV chỉ {num(tree_cv, 4)}. Random Forest "
         f"giảm phương sai nên tốt hơn hẳn cây đơn, đúng như lý thuyết. KNN trên "
         f"không gian one-hot nhiều chiều đạt PR-AUC {num(knn_pr, 4)} và ROC-AUC "
         f"{num(knn_roc, 4)} — thấp hơn nhóm boosting và Logistic Regression — và dự đoán "
         "chậm do phải tính khoảng cách tới toàn bộ tập train cho mỗi điểm (lời nguyền "
         "số chiều). LinearSVC tuy nhanh nhưng không cho xác suất trực tiếp nên chỉ "
         "mang tính đối chiếu.")
    # RQ2: class_weight
    def _dir(before, after):
        return ("tăng", "lên") if after > before else ("giảm", "xuống")
    for base, bal in [("LogReg", "LogReg_balanced"), ("Tree", "Tree_balanced"), ("RF", "RF_balanced")]:
        if base in cv0.index and bal in cv0.index:
            b, c = cv0.loc[base], cv0.loc[bal]
            rw, rp = _dir(b["recall_cv"], c["recall_cv"])
            pw, pp = _dir(b["precision_cv"], c["precision_cv"])
            body(f"So sánh {base} với {bal} (trả lời RQ2): Recall {rw} từ "
                 f"{num(b['recall_cv'], 4)} {rp} {num(c['recall_cv'], 4)}, Precision {pw} từ "
                 f"{num(b['precision_cv'], 4)} {pp} {num(c['precision_cv'], 4)}, F1 thay đổi từ "
                 f"{num(b['f1_cv'], 4)} thành {num(c['f1_cv'], 4)}, trong khi PR-AUC gần như "
                 f"không đổi ({num(b['pr_auc_cv'], 4)} so với {num(c['pr_auc_cv'], 4)}). Điều này "
                 "đúng như lý thuyết: class_weight chủ yếu dịch ngưỡng quyết định ngầm "
                 "chứ không thay đổi thứ hạng xác suất, nên các metric không phụ thuộc "
                 "ngưỡng gần như giữ nguyên.")
    h2("5.2", "Tinh chỉnh siêu tham số")
    best_params = load_json("best_params.json")
    rows = []
    for name in cvt.index:
        r = cvt.loc[name]
        before = float(cv0.loc[name, "pr_auc_cv"]) if name in cv0.index else float("nan")
        rows.append([name, json.dumps(best_params.get(name, {}), ensure_ascii=False),
                     f"{num(before, 4)}" if before == before else "—",
                     f"{num(r['pr_auc_cv'], 4)} ± {num(r['pr_auc_std'], 4)}",
                     f"{nums(r['pr_auc_cv'] - before, 4)}" if before == before else "—"])
    add_table(["Mô hình", "Tham số tốt nhất", "PR-AUC trước tuning", "PR-AUC sau tuning", "Cải thiện"],
              rows, widths=[2.0, 6.5, 2.2, 2.5, 1.8], font_size=10.5)
    para("Bảng 5.2. Kết quả tuning bằng RandomizedSearchCV (tối ưu PR-AUC, chỉ trên train).",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    body(f"Sau tuning, mô hình tốt nhất là {tuned_best['best_model']} với PR-AUC CV "
         f"{num(tuned_best['pr_auc_cv'], 4)} (độ lệch chuẩn {num(tuned_best['pr_auc_std'], 4)}).")
    figure("fig08_validation_curve_tree.png",
           "Hình 5.1. Validation curve của Decision Tree theo max_depth (PR-AUC).")
    body("Hình 5.1 cho thấy khi max_depth tăng, điểm train tăng liên tục trong khi "
         "điểm CV tăng rồi đi ngang/giảm — khoảng cách hai đường chính là overfitting. "
         "Điểm CV cao nhất rơi vào vùng max_depth vừa phải, khớp với tham số tốt nhất "
         "tìm được bằng RandomizedSearchCV.")
    figure("fig09_learning_curve_best.png",
           "Hình 5.2. Learning curve của mô hình tốt nhất (PR-AUC theo số mẫu train).")
    body("Learning curve cho thấy hai đường train và CV tiến lại gần nhau khi thêm dữ "
         "liệu, nghĩa là mô hình không bị overfitting nặng và việc thêm dữ liệu vẫn "
         "còn giúp ích đôi chút, nhưng lợi ích biên đang giảm dần.")
    h2("5.3", "Chọn ngưỡng và đánh giá cuối trên tập test")
    body(f"Áp dụng quy tắc chọn mô hình ở Chương 4, mô hình cuối là "
         f"{final['final_model']}. {final['selection_reason']} Ngưỡng tối ưu F1 chọn "
         f"từ dự đoán out-of-fold trên train là {num(final['threshold_f1'], 3)}; ngưỡng "
         f"giữ recall tối thiểu 0,7 là {num(final['threshold_recall70'], 3)}.")
    m = final["test_metrics_f1_threshold"]
    m70 = final["test_metrics_recall70_threshold"]
    ci = final["bootstrap_ci_95"]
    add_table(["Metric (test)", "Ngưỡng tối ưu F1", "Ngưỡng giữ recall ≥ 0,7"],
              [["ROC-AUC", f"{num(m['roc_auc'], 4)}", f"{num(m70['roc_auc'], 4)}"],
               ["PR-AUC", f"{num(m['pr_auc'], 4)}", f"{num(m70['pr_auc'], 4)}"],
               ["F1 (yes)", f"{num(m['f1'], 4)}", f"{num(m70['f1'], 4)}"],
               ["Recall (yes)", f"{num(m['recall'], 4)}", f"{num(m70['recall'], 4)}"],
               ["Precision (yes)", f"{num(m['precision'], 4)}", f"{num(m70['precision'], 4)}"],
               ["Accuracy", f"{num(m['accuracy'], 4)}", f"{num(m70['accuracy'], 4)}"],
               ["Confusion (TN/FP/FN/TP)", f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}",
                f"{m70['tn']}/{m70['fp']}/{m70['fn']}/{m70['tp']}"]],
              widths=[4.5, 5.0, 5.5], font_size=11.5)
    para(f"Bảng 5.3. Kết quả đánh giá cuối trên test cho mô hình "
         f"{final['final_model']}.", italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    body(f"Khoảng tin cậy bootstrap 95% trên test: ROC-AUC "
         f"[{num(ci['roc_auc'][0], 3)}, {num(ci['roc_auc'][1], 3)}], PR-AUC "
         f"[{num(ci['pr_auc'][0], 3)}, {num(ci['pr_auc'][1], 3)}], F1 "
         f"[{num(ci['f1'][0], 3)}, {num(ci['f1'][1], 3)}]. Khoảng tin cậy hẹp cho thấy kết quả "
         "ổn định, không phải do may mắn trên một mẫu test cụ thể.")
    figure("fig10_final_test_curves.png",
           "Hình 5.3. Đường cong ROC, PR và ma trận nhầm lẫn của mô hình cuối trên test.")
    body("Bảng 5.4 so sánh tất cả mô hình đã tuning trên test (mỗi mô hình dùng "
         "ngưỡng F1 riêng chọn từ train, chỉ để báo cáo; không dùng bảng này để đổi "
         "lựa chọn mô hình).")
    rows = []
    for name in test_table.index:
        r = test_table.loc[name]
        rows.append([name, f"{num(r['threshold_f1'], 3)}", f"{num(r['roc_auc'], 4)}",
                     f"{num(r['pr_auc'], 4)}", f"{num(r['f1'], 4)}", f"{num(r['recall'], 4)}",
                     f"{num(r['precision'], 4)}"])
    add_table(["Mô hình", "Ngưỡng F1", "ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"],
              rows, widths=[2.6, 2.0, 1.8, 1.8, 1.6, 1.6, 2.0], font_size=11)
    para("Bảng 5.4. Kết quả trên test của các mô hình đã tuning (chỉ để báo cáo).",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    best_test = test_table.sort_values("pr_auc", ascending=False).index[0]
    body(f"Trên test, mô hình dẫn đầu về PR-AUC là {best_test} "
         f"({num(test_table.loc[best_test, 'pr_auc'], 4)}), phù hợp với thứ hạng đã thấy "
         "trong CV — kết quả CV và test nhất quán với nhau, chứng tỏ quy trình chọn "
         "mô hình không bị rò rỉ và mô hình tổng quát hóa tốt.")


def chapter_6():
    import pandas as pd
    ana = load_json("analysis_summary.json")
    extra = load_json("extra_experiments.json")
    final = load_json("final_summary.json")

    h1("6", "PHÂN TÍCH, DIỄN GIẢI VÀ THÍ NGHIỆM BỔ SUNG")
    h2("6.1", "Phân tích lỗi theo nhóm khách hàng")
    body(f"Mô hình cuối ({final['final_model']}) không sai đều ở mọi nhóm khách. "
         "Bảng 6.1 liệt kê các nhóm có recall của lớp yes thấp nhất — tức là những "
         "khách đồng ý nhưng mô hình bỏ sót — cùng số lượng gọi nhầm (false positive) "
         "theo từng nhóm.")
    ebg = ana["error_by_group"]
    for group in ["job", "poutcome", "month"]:
        tbl = ebg[group]
        items = sorted(tbl.items(), key=lambda kv: kv[1]["recall"])[:5]
        rows = [[k, int(v["n_yes"]), f"{num(v['recall'], 3)}", int(v["n_fp"]), int(v["n_fn"])]
                for k, v in items]
        add_table([group, "Số khách yes", "Recall", "Số gọi nhầm (FP)", "Số bỏ sót (FN)"],
                  rows, widths=[3.0, 3.0, 2.5, 3.5, 3.0], font_size=11)
    para("Bảng 6.1. Các nhóm có recall thấp nhất (5 nhóm đầu của mỗi biến).",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    body("Nhìn chung, mô hình bỏ sót nhiều nhất ở nhóm blue-collar (recall 0,27) — "
         "nhóm có tỷ lệ đồng ý thấp nhất trong EDA — cùng các nhóm nghề hiếm như "
         "entrepreneur. Ngược lại, nhóm poutcome = success được bắt gần như trọn vẹn "
         "vì tín hiệu quá rõ. Chi tiết đầy đủ theo từng nhóm nằm trong các file "
         "error_by_*.csv kèm theo báo cáo.")
    h2("6.2", "Hệ số Logistic Regression và permutation importance")
    odds_top = ana["logreg_odds_top10"]
    odds_bot = ana["logreg_odds_bottom10"]
    body("Hình 6.1 cho thấy các đặc trưng làm tăng mạnh nhất khả năng đồng ý theo "
         "Logistic Regression (odds ratio). Hai hệ số lớn nhất là month_mar và "
         "cons.price.idx. Lưu ý: vì ba biến vĩ mô đa cộng tuyến mạnh, dấu và độ lớn "
         "của từng hệ số vĩ mô riêng lẻ không đáng tin — ví dụ emp.var.rate có odds "
         "ratio chỉ 0,095 trong khi nr.employed là 1,582, dù hai biến này tương quan "
         "0,91 — chỉ nên đọc xu hướng chung của nhóm. Đáng chú ý, poutcome = success "
         "không nằm trong top 10 odds ratio dù EDA cho thấy nhóm này đồng ý tới 66%: "
         "trong mô hình đa biến, tín hiệu của nó đã bị “hấp thụ” một phần bởi cờ "
         "contacted_before và các biến vĩ mô — một lời nhắc rằng hệ số hồi quy đo "
         "tác động biên khi các biến khác giữ nguyên, khác với tỷ lệ đơn biến trong "
         "EDA.")
    figure("fig11_logreg_odds.png",
           "Hình 6.1. Odds ratio của Logistic Regression (top/bottom 8).")
    imp = ana["permutation_importance_top10"]
    body("Permutation importance (Hình 6.2) đo mức giảm PR-AUC khi hoán vị từng biến "
         "gốc trên test. Biến quan trọng nhất là "
         f"{list(imp.keys())[0]} (giảm {num(list(imp.values())[0], 4)} PR-AUC), tiếp theo là "
         f"{list(imp.keys())[1]} và {list(imp.keys())[2]}. Đáng chú ý, duration có "
         "importance bằng 0 vì pipeline không dùng nó — bằng chứng trực quan rằng "
         "kịch bản A thực sự loại bỏ biến rò rỉ.")
    figure("fig12_permutation_importance.png",
           "Hình 6.2. Permutation importance trên test (giảm PR-AUC).")
    h2("6.3", "Cây nông minh họa luật kinh doanh")
    body("Hình 6.3 là cây quyết định nông (max_depth = 3) để đọc thành luật kinh "
         "doanh. Các nhánh đầu tiên tách theo kết quả chiến dịch trước và các chỉ số "
         "vĩ mô, cho thấy mô hình học được đúng các quy tắc mà chuyên viên "
         "telemarketing vẫn dùng theo kinh nghiệm: ưu tiên khách đã từng đồng ý và "
         "gọi vào thời điểm lãi suất thấp.")
    figure("fig13_shallow_tree.png",
           "Hình 6.3. Cây quyết định nông (max_depth = 3) minh họa luật.")
    if "shap_top12" in ana:
        h2("6.4", "SHAP")
        body("Hình 6.4 cho thấy |SHAP value| trung bình của 12 đặc trưng quan trọng "
             "nhất theo mô hình cuối. Thứ hạng SHAP nhất quán với permutation "
             "importance, củng cố kết luận về nhóm yếu tố quyết định.")
        figure("fig14_shap_summary.png",
               "Hình 6.4. SHAP — |SHAP value| trung bình (top 12).")
    h2("6.5", "Thí nghiệm duration: bằng chứng định lượng về leakage (RQ4)")
    ab = extra["duration_ab_cv"]
    a, b = ab["A_khong_duration"], ab["B_co_duration"]
    body(f"Kịch bản A (không duration) đạt ROC-AUC CV {num(a['roc_auc_cv'], 4)} và PR-AUC "
         f"CV {num(a['pr_auc_cv'], 4)}; kịch bản B (có duration) đạt ROC-AUC CV "
         f"{num(b['roc_auc_cv'], 4)} và PR-AUC CV {num(b['pr_auc_cv'], 4)}. Duration làm "
         f"PR-AUC tăng {nums(b['pr_auc_cv'] - a['pr_auc_cv'], 4)} và ROC-AUC tăng "
         f"{nums(b['roc_auc_cv'] - a['roc_auc_cv'], 4)}. Mức tăng này là “ảo”: duration "
         "phản ánh kết quả cuộc gọi (khách quan tâm thì nói chuyện lâu) chứ không phải "
         "thông tin có trước khi gọi. Mô hình B không thể dùng để quyết định ai nên "
         "được gọi, vì tại thời điểm ra quyết định, duration chưa tồn tại. Kết luận: "
         "chỉ kịch bản A mới là mô hình triển khai được; mọi con số của kịch bản B "
         "chỉ có giá trị minh họa.")
    h2("6.6", "Ablation theo nhóm biến (RQ3)")
    abl = extra["ablation_cv"]
    rows = [[k, f"{num(v['roc_auc_cv'], 4)}", f"{num(v['pr_auc_cv'], 4)}",
             f"{num(v['recall_cv'], 4)}", f"{num(v['f1_cv'], 4)}"] for k, v in abl.items()]
    add_table(["Nhóm biến", "ROC-AUC (CV)", "PR-AUC (CV)", "Recall", "F1"],
              rows, widths=[4.0, 2.8, 2.8, 2.2, 2.2], font_size=11)
    para("Bảng 6.2. Ablation theo nhóm biến (LogReg balanced, 5-Fold CV trên train).",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    all_pr = abl["All"]["pr_auc_cv"]
    best_single = max([(k, v["pr_auc_cv"]) for k, v in abl.items() if "+" not in k and k != "All"],
                      key=lambda kv: kv[1])
    body(f"Một mình nhóm {best_single[0]} đã đạt PR-AUC {num(best_single[1], 4)}, so với "
         f"{num(all_pr, 4)} khi dùng tất cả các nhóm. Nhóm chiến dịch và nhóm vĩ mô đóng "
         "góp nhiều hơn hẳn nhóm thông tin cá nhân — phù hợp với EDA: lịch sử liên hệ "
         "và thời điểm gọi quyết định nhiều hơn tuổi hay nghề nghiệp. Lưu ý: nhóm vĩ "
         "mô có thể “ăn” tín hiệu thời gian (cùng kỳ chiến dịch), nên cần thận trọng "
         "khi kết luận nhân quả; thí nghiệm 6.7 sẽ kiểm tra điểm này.")
    h2("6.7", "Chia theo thời gian và giả thuyết “đồng hồ thời gian”")
    ts = extra["time_split"]
    body(f"Khi huấn luyện trên 80% đầu (theo thứ tự thời gian) và đánh giá trên 20% "
         f"cuối — nơi tỷ lệ yes là {pct(ts['val_yes_rate'])} so với 6,38% ở phần đầu — "
         "hiệu năng giảm so với chia ngẫu nhiên. Đây là concept drift: thị trường và "
         "lãi suất thay đổi trong giai đoạn 2008–2010, nên mô hình học theo thời kỳ sẽ "
         "kém tổng quát hóa. Lưu ý: vì tỷ lệ yes ở hai phần khác nhau rất xa, PR-AUC "
         "không so sánh thẳng được giữa hai cách chia; ROC-AUC là thước đo so sánh "
         "chính ở đây.")
    rows = [[k, f"{num(v['roc_auc'], 4)}", f"{num(v['pr_auc'], 4)}"] for k, v in ts.items()
              if isinstance(v, dict)]
    add_table(["Mô hình (time split)", "ROC-AUC", "PR-AUC"], rows,
              widths=[6.0, 4.5, 4.5], font_size=11)
    para("Bảng 6.3. Kết quả trên time split (train 80% đầu, test 20% cuối).",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    fn = final["final_model"]
    full_roc = ts[fn + "_full"]["roc_auc"]
    nm_roc = ts[fn + "_no_macro"]["roc_auc"]
    if nm_roc > full_roc:
        body(f"Điểm nhấn của thí nghiệm: khi bỏ nhóm biến vĩ mô, ROC-AUC trên time "
             f"split TĂNG từ {num(full_roc, 4)} lên {num(nm_roc, 4)}. Điều này ủng hộ giả thuyết "
             "của giảng viên rằng các biến vĩ mô (euribor3m, nr.employed, emp.var.rate) "
             "thực chất là “đồng hồ thời gian”: chúng mã hóa thời kỳ thay vì đặc điểm "
             "khách hàng, nên mô hình dựa vào chúng sẽ học theo thời kỳ và mất hiệu "
             "năng khi thời gian trôi qua. Với dữ liệu mới ngoài giai đoạn huấn luyện, "
             "bản không dùng biến vĩ mô có thể bền vững hơn — một hướng đáng cân nhắc "
             "cho triển khai thực tế.")
    else:
        body(f"Khi bỏ nhóm biến vĩ mô, ROC-AUC trên time split là {num(nm_roc, 4)} so với "
             f"{num(full_roc, 4)} của bản đầy đủ. Như vậy giả thuyết “đồng hồ thời gian” "
             "không được ủng hộ mạnh trong thí nghiệm này: biến vĩ mô vẫn mang thông "
             "tin hữu ích ngay cả khi đánh giá ngoài thời gian. Tuy nhiên, hiệu năng "
             "trên time split vẫn thấp hơn chia ngẫu nhiên, cho thấy concept drift là "
             "có thật và mô hình cần được huấn luyện lại định kỳ khi triển khai.")
    h2("6.8", "Hiệu chỉnh xác suất (calibration)")
    cal = extra["calibration"]
    body(f"Vì mô hình cuối dùng class_weight, xác suất đầu ra bị lệch lên so với tỷ lệ "
         f"thật. Hiệu chỉnh isotonic giảm Brier score trên test từ "
         f"{num(cal['brier_before'], 4)} xuống {num(cal['brier_after'], 4)}. Nếu dùng xác suất "
         "làm số liệu kinh doanh (ví dụ kỳ vọng doanh thu), nên dùng bản đã hiệu "
         "chỉnh; lưu ý sau calibration thì ngưỡng phải chọn lại bằng out-of-fold.")
    figure("fig15_calibration_curve.png",
           "Hình 6.5. Calibration curve trước và sau hiệu chỉnh isotonic (test).")
    h2("6.9", "Gain/Lift và chi phí–lợi ích (RQ5)")
    top20 = extra["top20_capture"]
    body(f"Gọi top 20% khách theo điểm mô hình sẽ bắt được {pct(top20)} số người đồng "
         "ý — trả lời trực tiếp RQ5. Đường gain chart (Hình 6.6) cho thấy mô hình "
         "vượt xa đường ngẫu nhiên: càng gọi ít, lợi thế càng lớn.")
    dec = extra["decile_table"]
    rows = [[d, v["n_called"], int(v["n_yes"]), f"{num(v['cum_capture'], 3)}", num(v['lift'], 2)]
            for d, v in sorted(dec.items(), key=lambda kv: int(kv[0]))]
    add_table(["Decile", "Số khách gọi", "Số yes", "Tỷ lệ bắt tích lũy", "Lift"],
              rows, widths=[2.0, 3.5, 2.5, 4.0, 3.0], font_size=11)
    para("Bảng 6.4. Bảng decile: khách xếp theo điểm mô hình, gọi từ decile 1 xuống.",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    figure("fig16_gain_chart.png",
           "Hình 6.6. Gain chart trên test (mô hình cuối so với gọi ngẫu nhiên).")
    cb = extra["cost_benefit"]
    rows = [[k, v["top_pct"], v["capture_pct"], v["profit_max"], v["profit_call_all"]]
            for k, v in cb.items()]
    add_table(["Giả định (C, R)", "Gọi top (%)", "Bắt được (%)", "Lợi nhuận tối đa", "Lợi nhuận gọi tất cả"],
              rows, widths=[3.0, 2.5, 2.5, 3.5, 3.5], font_size=11)
    para("Bảng 6.5. Chi phí–lợi ích với các cặp giả định (C, R). C = chi phí một cuộc "
         "gọi, R = lợi nhuận khi một khách đồng ý. Đây là giả định của nhóm, không "
         "phải số thật của ngân hàng.", italic=True, size=11.5,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    body("Bảng 6.5 cho thấy ngưỡng gọi tối ưu thay đổi theo cặp (C, R): chi phí gọi "
         "càng cao hoặc giá trị khách càng thấp thì càng nên gọi ít hơn. Ngân hàng có "
         "thể thay số thật của mình vào công thức này để ra quyết định.")
    h2("6.10", "SMOTE so với class_weight")
    if "smote_vs_class_weight" in extra:
        sm = extra["smote_vs_class_weight"]
        rows = [[k, f"{num(v['roc_auc_cv'], 4)}", f"{num(v['pr_auc_cv'], 4)}",
                 f"{num(v['f1_cv'], 4)}", f"{num(v['recall_cv'], 4)}"] for k, v in sm.items()]
        add_table(["Phương pháp", "ROC-AUC (CV)", "PR-AUC (CV)", "F1", "Recall"],
                  rows, widths=[4.5, 2.8, 2.8, 2.2, 2.2], font_size=11)
        para("Bảng 6.6. SMOTE so với class_weight (LogReg, 5-Fold CV trên train; SMOTE "
             "chỉ lấy mẫu lại trong fold train).", italic=True, size=11.5,
             align=WD_ALIGN_PARAGRAPH.CENTER)
        body("SMOTE không vượt trội so với class_weight mà còn tốn thời gian hơn và làm "
             "thay đổi phân phối dữ liệu huấn luyện. Vì vậy đồ án giữ class_weight làm "
             "phương pháp chính, đúng như quyết định ở Chương 4.")
    else:
        body("Thí nghiệm SMOTE không chạy được trong môi trường thực nghiệm của nhóm "
             f"({extra.get('smote_status', '')}). class_weight vẫn là phương pháp chính "
             "và đã đủ cho bài toán này.")
    h2("6.11", "Đối chiếu số liệu tham chiếu về Random Forest")
    body("Trong quá trình thực hiện, giảng viên cung cấp bảng số liệu chạy lại với "
         "Random Forest balanced đạt ROC-AUC 0,814 và PR-AUC 0,482 trên test, cao hơn "
         "hẳn các cấu hình RF mà nhóm thử được (RF 100 cây balanced: ROC-AUC 0,783, "
         "PR-AUC 0,389; RF 300 cây balanced_subsample: ROC-AUC 0,783, PR-AUC 0,426; "
         "RF 300 cây balanced: ROC-AUC 0,784, PR-AUC 0,388). Vì Logistic Regression và "
         "HistGradientBoosting của nhóm khớp gần như tuyệt đối với số liệu tham chiếu "
         "của thầy (sai lệch dưới 0,001), vấn đề nằm riêng ở cấu hình RF chứ không "
         "phải dữ liệu hay tiền xử lý chung. Thí nghiệm tuning ở Chương 5 đã giải "
         "đáp câu hỏi này: Random Forest sau khi tinh chỉnh (400 cây, max_depth = 8, "
         "min_samples_leaf = 5, max_features = 0,3) đạt PR-AUC CV 0,4684, tăng 0,0536 "
         "so với bản mặc định 0,4148 và trở thành mô hình tốt nhất sau tuning. Phần "
         "lớn chênh lệch với số của thầy đến từ việc RF mặc định dùng cây không giới "
         "hạn độ sâu (overfit nặng: PR-AUC train 0,999 so với CV 0,415) và không giới "
         "hạn số đặc trưng mỗi lần tách. Trên tập test, RF đã tuning đạt "
         f"ROC-AUC {num(final['test_metrics_f1_threshold']['roc_auc'], 4)} và PR-AUC "
         f"{num(final['test_metrics_f1_threshold']['pr_auc'], 4)} — gần như trùng khớp với "
         "số tham chiếu của thầy (0,814 / 0,482). Như vậy RF của thầy gần như chắc "
         "chắn cũng là bản đã tinh chỉnh; F1/Recall còn chênh đôi chút "
         f"({num(final['test_metrics_f1_threshold']['f1'], 4)}/"
         f"{num(final['test_metrics_f1_threshold']['recall'], 4)} của nhóm so với "
         "0,508/0,63 của thầy) chủ yếu do ngưỡng quyết định khác nhau — nhóm dùng "
         "ngưỡng tối ưu F1 chọn từ out-of-fold. Thí nghiệm SMOTE ở mục 6.10 là mảnh "
         "ghép cuối để kiểm tra giả thuyết resampling. Dù kết quả thế nào, mô hình "
         "cuối của đồ án "
         f"({final['final_model']}, PR-AUC test {num(final['test_metrics_f1_threshold']['pr_auc'], 4)}) "
         "vẫn đứng ở mức hiệu năng tương đương trên test, nên kết luận chung của đồ "
         "án không phụ thuộc vào việc tái lập đúng một con số RF cụ thể. Nhóm ghi "
         "nhận điểm chưa khớp này một cách minh bạch thay vì điều chỉnh số liệu cho "
         "khớp.")
    h2("6.12", "Hạn chế")
    bullet("Dữ liệu chỉ từ một ngân hàng Bồ Đào Nha trong giai đoạn 2008–2010, ngay "
           "sau khủng hoảng tài chính; hành vi khách hàng có thể đã thay đổi.")
    bullet("Tập test có tỷ lệ yes khác xa train trong cách chia theo thời gian "
           "(30,83% so với 6,38%), nên PR-AUC không so sánh thẳng được giữa hai cách "
           "chia; đồ án đã dùng ROC-AUC để đối chiếu và nêu rõ lý do.")
    bullet("Các con số chi phí–lợi ích dùng giả định của nhóm (C = 1, R = 15), không "
           "phải số thật của ngân hàng.")
    bullet("Mô hình chưa dùng thông tin hành vi khách hàng ngoài chiến dịch "
           "(ví dụ số dư tài khoản thực — biến balance không có trong bộ "
           "bank-additional), và chưa thử các mô hình chuỗi thời gian.")


def chapter_7():
    import pandas as pd
    final = load_json("final_summary.json")
    extra = load_json("extra_experiments.json")
    h1("7", "KẾT LUẬN")
    m = final["test_metrics_f1_threshold"]
    body(f"Mô hình cuối ({final['final_model']}) đạt ROC-AUC "
         f"{num(m['roc_auc'], 4)}, PR-AUC {num(m['pr_auc'], 4)}, F1 {num(m['f1'], 4)}, Recall "
         f"{num(m['recall'], 4)} và Precision {num(m['precision'], 4)} trên tập test (ngưỡng "
         f"tối ưu F1 = {num(final['threshold_f1'], 3)}). Toàn bộ quy trình — từ chia dữ "
         "liệu, tiền xử lý, so sánh, tuning, chọn ngưỡng đến đánh giá cuối — tuân thủ "
         "tám nguyên tắc vàng của đề và có thể chạy lại hoàn toàn bằng code trong repo.")
    h2("7.1", "Trả lời câu hỏi nghiên cứu")
    bullet(f"Mô hình {final['final_model']} cân bằng tốt nhất giữa Recall và "
           f"Precision của lớp yes (xem Bảng 5.3–5.4).",
           bold_prefix="RQ1: ")
    bullet("class_weight làm Recall tăng mạnh và Precision giảm, trong khi PR-AUC và "
           "ROC-AUC gần như không đổi — vì nó chỉ dịch ngưỡng ngầm chứ không đổi thứ "
           "hạng xác suất.", bold_prefix="RQ2: ")
    bullet("Nhóm chiến dịch (đặc biệt là poutcome và contacted_before) và nhóm vĩ mô "
           "đóng góp nhiều nhất; nhóm thông tin cá nhân đóng góp ít nhất.",
           bold_prefix="RQ3: ")
    bullet("Duration làm PR-AUC tăng vọt nhưng giá trị đó là ảo vì duration chỉ biết "
           "sau khi gọi; chỉ kịch bản không duration mới triển khai được.",
           bold_prefix="RQ4: ")
    bullet(f"Gọi top 20% khách theo điểm mô hình bắt được "
           f"{pct(extra['top20_capture'])} số người đồng ý.",
           bold_prefix="RQ5: ")
    h2("7.2", "Hướng phát triển")
    bullet("Huấn luyện lại mô hình định kỳ (theo quý) để chống concept drift, và theo "
           "dõi hiệu năng trên dữ liệu mới bằng time split.")
    bullet("Thử nghiệm mô hình không dùng biến vĩ mô cho triển khai dài hạn, nếu giả "
           "thuyết “đồng hồ thời gian” được củng cố thêm.")
    bullet("Bổ sung dữ liệu hành vi khách hàng (số dư, lịch sử giao dịch) nếu ngân hàng "
           "cung cấp, và thử mô hình hiệu chỉnh xác suất cho bài toán doanh thu.")
    bullet("Triển khai demo thành ứng dụng nội bộ cho đội telemarketing, với danh sách "
           "khách xếp hạng cập nhật hằng ngày.")
    h2("7.3", "Khai báo sử dụng AI")
    body("Nhóm có sử dụng trợ lý AI (Muse) để hỗ trợ viết code khung, kiểm chứng số "
         "liệu và soạn thảo báo cáo. Mọi con số trong báo cáo đều do code trong repo "
         "chạy ra và được kiểm chứng lại trước khi đưa vào; các quyết định phương pháp "
         "(tiền xử lý, metric, ngưỡng, mô hình cuối) do nhóm tự chốt theo Cẩm nang "
         "thực hiện dự án và phản hồi của giảng viên.")


def references_and_appendix():
    h1("", "TÀI LIỆU THAM KHẢO")
    refs = [
        "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict "
        "the success of bank telemarketing. Decision Support Systems, 62, 22–31.",
        "UCI Machine Learning Repository. Bank Marketing Data Set. "
        "https://archive.ics.uci.edu/dataset/222/bank+marketing "
        "(DOI 10.24432/C5K306).",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
        "Discovery and Data Mining, 785–794.",
        "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. "
        "(2017). LightGBM: A highly efficient gradient boosting decision tree. Advances "
        "in Neural Information Processing Systems, 30.",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
        "predictions. Advances in Neural Information Processing Systems, 30.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal "
        "of Machine Learning Research, 12, 2825–2830.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        run = p.add_run(f"[{i}] {r}")
        run.font.name = FONT
        run.font.size = Pt(BODY_SIZE)
    doc.add_page_break()
    h1("", "PHỤ LỤC A. THAM SỐ TỐT NHẤT SAU TUNING")
    best_params = load_json("best_params.json")
    rows = [[k, json.dumps(v, ensure_ascii=False)] for k, v in best_params.items()]
    add_table(["Mô hình", "Tham số tốt nhất"], rows, widths=[3.0, 12.0], font_size=11)
    h1("", "PHỤ LỤC B. NHẬT KÝ THỰC NGHIỆM")
    body("Nhật ký đầy đủ (experiments/experiment_log.csv) ghi lại mọi thực nghiệm "
         "chính của đồ án kèm timestamp. Các bảng kết quả chi tiết theo từng giai đoạn "
         "nằm trong thư mục reports/ (cv_default_models.csv, cv_tuned_models.csv, "
         "test_results.csv, ablation_cv.csv, duration_ab_cv.csv, decile_table.csv, "
         "tuning_cv_results_*.csv).")
    h1("", "PHỤ LỤC C. HƯỚNG DẪN CHẠY LẠI")
    body("Cài đặt môi trường: pip install -r requirements.txt (Python 3.12). Chạy "
         "lần lượt: scripts/01_eda.py, scripts/02_baseline_and_cv.py, "
         "scripts/03_tuning.py, scripts/04_final_evaluation.py, "
         "scripts/05_analysis.py, scripts/06_extra_experiments.py. Mọi kết quả trong "
         "báo cáo đều được tái tạo từ các script này với SEED = 42. Demo: streamlit "
         "run app/app.py.")


def main():
    eda = load_json("eda_summary.json")
    cover_and_front()
    chapter_1(eda)
    chapter_2()
    chapter_3(eda)
    chapter_4()
    chapter_5()
    chapter_6()
    chapter_7()
    references_and_appendix()
    doc.save(str(OUT))
    print("Đã lưu", OUT)


if __name__ == "__main__":
    main()
