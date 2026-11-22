# -*- coding: utf-8 -*-
"""Sinh 7 notebook .ipynb từ các script giai đoạn (cùng một code, cùng kết quả).

Notebook 01-05 là luồng stratified (v1, giữ làm tham chiếu "cận trên lạc quan");
notebook 06-07 là luồng theo thời gian (v2, thiết kế chính sau góp ý 2026-11-14).
Notebook được SINH TỪ script (không viết tay) — xem README mục "Notebook".

Mỗi script scripts/0X_*.py -> notebooks/0X_*.ipynb:
- Phần module-level của script (import, hằng số đường dẫn) -> code cell đầu,
  với `Path(__file__)` đổi thành `Path("..")` vì notebook chạy trong thư mục
  notebooks/ (thư mục cha chính là gốc dự án).
- Thân hàm main() được dedent về top-level và tách thành các code cell theo
  các dòng tiêu đề dạng '# --- ... ---' (thành markdown cell).
- Bỏ dòng `if __name__ == "__main__": main()` vì thân main đã chạy trực tiếp.

Sau khi sinh, chạy `jupyter nbconvert --to notebook --execute` để kiểm chứng
notebook chạy lại được và cho ra đúng kết quả như script.
"""
import ast
import re
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ("01_eda", "01_eda"),
    ("02_baseline_and_cv", "02_baseline_and_cv"),
    ("03_tuning", "03_tuning"),
    ("04_final_evaluation", "04_final_evaluation"),
    ("05_extra_experiments", "05_extra_experiments"),
    ("06_time_design_tuning", "06_time_design_tuning"),
    ("07_time_final_evaluation", "07_time_final_evaluation"),
]
# Notebook 05 gộp cả phân tích lỗi (05_analysis) và thí nghiệm bổ sung (06_extra)
EXTRA = {"05_extra_experiments": ["scripts/05_analysis.py", "scripts/06_extra_experiments.py"],
         "06_time_design_tuning": ["scripts/07_time_design.py"],
         "07_time_final_evaluation": ["scripts/08_calibration_threshold.py",
                                      "scripts/09_final_time_evaluation.py",
                                      "scripts/10_time_extras.py"]}


def _node_source(lines, node):
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def split_script(text):
    """Tách script thành danh sách (kind, content)."""
    lines = text.split("\n")
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    cells = []
    if doc:
        cells.append(("md", doc.strip()))
    main_body = None
    header_lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            body_lines = lines[node.body[0].lineno - 1:node.body[-1].end_lineno]
            main_body = textwrap.dedent("\n".join(body_lines))
            continue
        if isinstance(node, ast.If):
            # bỏ `if __name__ == "__main__": main()`
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue  # docstring đã lấy
        header_lines.append(_node_source(lines, node))
    header = "\n".join(header_lines)
    # Notebook chạy trong thư mục notebooks/ -> gốc dự án là ".."
    header = header.replace('Path(__file__).resolve().parents[1]', 'Path("..").resolve()')
    header = header.replace("Path(__file__).resolve().parent", 'Path("..").resolve()')
    if "__file__" in header:
        raise ValueError("Còn tham chiếu __file__ chưa xử lý trong phần đầu script")
    if header.strip():
        cells.append(("code", header.rstrip()))
    if main_body is None:
        return cells
    buf = []
    for line in main_body.split("\n"):
        if re.match(r"^\s*#\s*-{3,}", line):
            if buf and "\n".join(buf).strip():
                cells.append(("code", "\n".join(buf).rstrip()))
                buf = []
            title = line.lstrip("# -").strip()
            cells.append(("md", "## " + title if title else "---"))
            continue
        buf.append(line)
    if buf and "\n".join(buf).strip():
        cells.append(("code", "\n".join(buf).rstrip()))
    return cells


def build_notebook(script_paths, title):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\nĐồ án Máy học — Bank Marketing (Nhóm 14). "
                                             "Chạy theo thứ tự từ trên xuống dưới."))
    for sp in script_paths:
        text = (ROOT / sp).read_text(encoding="utf-8")
        for kind, content in split_script(text):
            if not content.strip():
                continue
            if kind == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(content))
            else:
                nb.cells.append(nbf.v4.new_code_cell(content))
    return nb


def main():
    titles = {
        "01_eda": "01 — Hiểu dữ liệu và EDA",
        "02_baseline_and_cv": "02 — Mô hình baseline và so sánh bằng CV",
        "03_tuning": "03 — Tinh chỉnh siêu tham số",
        "04_final_evaluation": "04 — Chọn ngưỡng và đánh giá cuối trên test",
        "05_extra_experiments": "05 — Phân tích lỗi, diễn giải và thí nghiệm bổ sung",
        "06_time_design_tuning": "06 — Thiết kế theo thời gian và tuning công bằng (v2)",
        "07_time_final_evaluation": "07 — Calibrate, đánh giá cuối và thí nghiệm theo thời gian (v2)",
    }
    for name, _ in SCRIPTS:
        paths = EXTRA.get(name, [f"scripts/{name}.py"])
        nb = build_notebook(paths, titles[name])
        out = ROOT / "notebooks" / f"{name}.ipynb"
        nbf.write(nb, out)
        print(f"wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
