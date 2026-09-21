import win32com.client
import os
import sys

sys.path.insert(0, os.path.abspath("scripts"))
from ieee_paper_content import (
    PAPER_METADATA,
    SECTIONS,
    TABLES_DATA,
    REFERENCES_LIST,
    ACKNOWLEDGMENT_TEXT
)

def build_paper():
    base_dir = os.path.abspath(r"c:\Users\HP\Desktop\MCA Project\MCA Project AI Breed")
    template_path = os.path.abspath(r"C:\Users\HP\Downloads\conference-template-a4.docx")
    out_docx = os.path.join(base_dir, "IEEE_Project_Paper.docx")
    out_pdf = os.path.join(base_dir, "IEEE_Project_Paper.pdf")

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    doc = word.Documents.Open(template_path)

    # 1. Section 1 (Title)
    sec1 = doc.Sections(1)
    r1 = sec1.Range
    r1.End = r1.End - 1
    r1.Text = PAPER_METADATA["title"] + "\r"
    r1.Paragraphs(1).Range.Style = doc.Styles("paper title")

    # 2. Section 2 (Authors)
    sec2 = doc.Sections(2)
    sec2.PageSetup.TextColumns.SetCount(2)
    sec2.PageSetup.TextColumns.Spacing = 36.0
    a1 = PAPER_METADATA["authors"][0]
    a2 = PAPER_METADATA["authors"][1]
    r2 = sec2.Range
    r2.End = r2.End - 1
    r2.Text = (
        f"{a1['name']}\r{a1['dept']}\r{a1['institution']}\r{a1['city_country']}\r{a1['email']}\r"
        f"\x0e"
        f"{a2['name']}\r{a2['dept']}\r{a2['institution']}\r{a2['city_country']}\r{a2['email']}\r"
    )

    for p in r2.Paragraphs:
        p.Range.Style = doc.Styles("Author")
        p.Range.ParagraphFormat.Alignment = 1
        p.Range.ParagraphFormat.SpaceBefore = 0
        p.Range.ParagraphFormat.SpaceAfter = 1
        txt = p.Range.Text.strip()
        if txt in [a1["name"], a2["name"]]:
            p.Range.Font.Bold = True
            p.Range.Font.Size = 10.5
        elif "@" in txt:
            p.Range.Font.Italic = True
            p.Range.Font.Size = 8.5
        else:
            p.Range.Font.Size = 9

    # 3. Section 3 (Clear)
    sec3 = doc.Sections(3)
    r3 = sec3.Range
    r3.End = r3.End - 1
    r3.Text = ""

    # 4. Section 4 (Paper Body)
    sec4 = doc.Sections(4)
    r = sec4.Range
    r.End = r.End - 1
    r.Text = ""

    def add_paragraph(text, style_name, sb=0, sa=3):
        r.Collapse(0)
        start_pos = r.End
        r.InsertAfter(text + "\r")
        end_pos = r.End
        p_rng = doc.Range(start_pos, end_pos)
        p_rng.Style = doc.Styles(style_name)
        p_rng.ParagraphFormat.SpaceBefore = sb
        p_rng.ParagraphFormat.SpaceAfter = sa
        p_rng.ParagraphFormat.KeepWithNext = False
        r.SetRange(end_pos, end_pos)
        return p_rng

    def insert_ieee_table(table_data):
        add_paragraph(f"TABLE {table_data['num']}", "table head", sb=8, sa=1)
        add_paragraph(table_data["title"], "table head", sb=0, sa=4)

        r.Collapse(0)
        headers = table_data["headers"]
        rows = table_data["rows"]
        tbl = doc.Tables.Add(r, len(rows) + 1, len(headers))
        tbl.Style = "Table Normal"
        tbl.Borders.Enable = False
        tbl.Borders(-1).LineStyle = 1
        tbl.Borders(-1).LineWidth = 4
        tbl.Borders(-3).LineStyle = 1
        tbl.Borders(-3).LineWidth = 4
        tbl.Rows(1).Borders(-3).LineStyle = 1
        tbl.Rows(1).Borders(-3).LineWidth = 4

        col_widths = table_data.get("col_widths")
        for c_idx, h in enumerate(headers, start=1):
            cell = tbl.Cell(1, c_idx)
            cell.Range.Text = h
            cell.Range.Font.Name = "Times New Roman"
            cell.Range.Font.Size = 7.5
            cell.Range.Font.Bold = True
            cell.Range.ParagraphFormat.Alignment = 1
            if col_widths and c_idx <= len(col_widths):
                cell.Width = col_widths[c_idx - 1]

        for r_idx, row in enumerate(rows, start=2):
            for c_idx, val in enumerate(row, start=1):
                cell = tbl.Cell(r_idx, c_idx)
                cell.Range.Text = str(val)
                cell.Range.Font.Name = "Times New Roman"
                cell.Range.Font.Size = 7.5
                cell.Range.ParagraphFormat.Alignment = 0 if c_idx == 1 else 1
                if col_widths and c_idx <= len(col_widths):
                    cell.Width = col_widths[c_idx - 1]

        r.SetRange(tbl.Range.End, tbl.Range.End)
        p_sp = doc.Paragraphs.Add(r)
        p_sp.Range.Text = "\r"
        p_sp.Range.Font.Size = 4
        r.SetRange(p_sp.Range.End, p_sp.Range.End)

    def insert_figure(img_rel_path, caption_text, width_pt=238):
        img_abs = os.path.join(base_dir, img_rel_path)
        if not os.path.exists(img_abs):
            return

        r.Collapse(0)
        p_img = doc.Paragraphs.Add(r)
        p_img.Range.ParagraphFormat.Alignment = 1
        shape = doc.InlineShapes.AddPicture(img_abs, False, True, p_img.Range)
        shape.Width = width_pt
        shape.LockAspectRatio = True
        r.SetRange(p_img.Range.End, p_img.Range.End)

        # Do not prepend 'Fig. X.' because the style figure caption does it automatically!
        # Check if caption starts with 'Fig. X.  '
        clean_cap = caption_text
        if clean_cap.startswith("Fig. "):
            parts = clean_cap.split(".  ", 1)
            if len(parts) > 1:
                clean_cap = parts[1]
            else:
                parts2 = clean_cap.split(". ", 1)
                if len(parts2) > 1:
                    clean_cap = parts2[1]

        add_paragraph(clean_cap, "figure caption", sb=2, sa=6)

    def insert_equation(eq_text, eq_num_str):
        eq_line = f"\t{eq_text}\t{eq_num_str}"
        p_eq = add_paragraph(eq_line, "equation", sb=3, sa=3)
        p_eq.Font.Name = "Times New Roman"
        p_eq.Font.Size = 9

    # Abstract
    p_abs = add_paragraph("Abstract\u2014" + PAPER_METADATA["abstract"], "Abstract", sb=6, sa=4)
    r_lead = doc.Range(p_abs.Start, p_abs.Start + 9)
    r_lead.Bold = True
    r_lead.Italic = True

    # Keywords
    p_kw = add_paragraph("Keywords\u2014" + PAPER_METADATA["keywords"], "Keywords", sb=2, sa=8)
    r_kw = doc.Range(p_kw.Start, p_kw.Start + 9)
    r_kw.Bold = True
    r_kw.Italic = True

    # Section I
    sec_i = SECTIONS[0]
    add_paragraph(sec_i["title"], "Heading 1", sb=8, sa=3)
    for p in sec_i["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=3)

    # Section II
    sec_ii = SECTIONS[1]
    add_paragraph(sec_ii["title"], "Heading 1", sb=8, sa=3)
    for p in sec_ii["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=3)

    # Section III
    sec_iii = SECTIONS[2]
    add_paragraph(sec_iii["title"], "Heading 1", sb=8, sa=3)
    # Intro paragraph
    add_paragraph(sec_iii["paragraphs"][0], "Body Text", sb=0, sa=3)
    # Sub A
    sub_a = sec_iii["paragraphs"][1].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=3)
    add_paragraph(sec_iii["paragraphs"][2], "Body Text", sb=0, sa=3)
    # Sub B
    sub_b = sec_iii["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=3)
    # Sub C
    sub_c = sec_iii["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=3)
    insert_ieee_table(TABLES_DATA[0]) # Table I
    # Sub D
    sub_d = sec_iii["paragraphs"][5].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=3)

    # Section IV
    sec_iv = SECTIONS[3]
    add_paragraph(sec_iv["title"], "Heading 1", sb=8, sa=3)
    # Sub A
    sub_a = sec_iv["paragraphs"][0].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=3)
    add_paragraph(sec_iv["paragraphs"][1], "Body Text", sb=0, sa=3)
    insert_figure(
        "documentation_assets/figures/workflow_diagram.png",
        "Proposed end-to-end intelligent breed classification pipeline."
    )
    # Sub B
    sub_b = sec_iv["paragraphs"][2].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=3)
    # Sub C
    sub_c = sec_iv["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=3)
    insert_equation("L_{LS}(y, \\hat{y}) = -(1 - \\epsilon)\\sum_{k=1}^K y_k \\log \\hat{y}_k - \\frac{\\epsilon}{K}\\sum_{k=1}^K \\log \\hat{y}_k", "(1)")
    # Sub D
    sub_d = sec_iv["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=3)
    insert_equation("L_{Grad-CAM}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)", "(2)")

    # Section V
    sec_v = SECTIONS[4]
    add_paragraph(sec_v["title"], "Heading 1", sb=8, sa=3)
    for p in sec_v["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=3)

    # Section VI
    sec_vi = SECTIONS[5]
    add_paragraph(sec_vi["title"], "Heading 1", sb=8, sa=3)
    # Sub A
    sub_a = sec_vi["paragraphs"][0].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=3)
    # Sub B
    sub_b = sec_vi["paragraphs"][1].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=3)
    insert_ieee_table(TABLES_DATA[1]) # Table II
    # Sub C
    sub_c = sec_vi["paragraphs"][2].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=3)
    insert_ieee_table(TABLES_DATA[2]) # Table III
    # Sub D
    sub_d = sec_vi["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=3)
    insert_figure(
        "documentation_assets/charts/v2_training_curves.png",
        "Training and validation loss and accuracy convergence trajectories across 50 epochs."
    )
    insert_figure(
        "documentation_assets/charts/confusion_matrix.png",
        "Normalized confusion matrix across 82 indigenous Indian livestock classes on the locked test partition."
    )
    # Sub E
    sub_e = sec_vi["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_e[0].replace("E. ", ""), "Heading 2", sb=5, sa=2)
    add_paragraph(sub_e[1], "Body Text", sb=0, sa=3)
    insert_figure(
        "documentation_assets/figures/fig4_gradcam_composite.png",
        "Grad-CAM visual interpretability overlays for Gir cattle (top) and Bhadawari buffalo (bottom)."
    )

    # Section VII
    sec_vii = SECTIONS[6]
    add_paragraph(sec_vii["title"], "Heading 1", sb=8, sa=3)
    for p in sec_vii["paragraphs"]:
        if any(p.startswith(f"{c}. ") for c in ["A", "B", "C"]):
            lines = p.split("\n", 1)
            add_paragraph(lines[0][3:], "Heading 2", sb=5, sa=2)
            if len(lines) > 1:
                add_paragraph(lines[1], "Body Text", sb=0, sa=3)
        else:
            add_paragraph(p, "Body Text", sb=0, sa=3)

    # Section VIII
    sec_viii = SECTIONS[7]
    add_paragraph(sec_viii["title"], "Heading 1", sb=8, sa=3)
    for p in sec_viii["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=3)

    # Acknowledgment (Heading 5 per template)
    add_paragraph("ACKNOWLEDGMENT", "Heading 5", sb=8, sa=3)
    add_paragraph(ACKNOWLEDGMENT_TEXT, "Body Text", sb=0, sa=3)

    # References (Heading 5 per template)
    add_paragraph("REFERENCES", "Heading 5", sb=8, sa=3)
    for ref_text in REFERENCES_LIST:
        p_r = add_paragraph(ref_text, "references", sb=0, sa=2)
        p_r.Font.Name = "Times New Roman"
        p_r.Font.Size = 8

    doc.SaveAs2(out_docx, 16)
    doc.SaveAs2(out_pdf, 17)

    pages = doc.ComputeStatistics(2)
    print(f"Final IEEE Paper Generated: {pages} Pages.")
    doc.Close(False)
    word.Quit()
    return True

if __name__ == "__main__":
    build_paper()
