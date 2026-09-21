"""
Generate complete IEEE-style research paper from official A4 conference template.
Outputs:
- IEEE_Project_Paper.docx
- IEEE_Project_Paper.pdf
"""

import os
import sys
import win32com.client
from ieee_paper_content import (
    PAPER_METADATA,
    SECTIONS,
    TABLES_DATA,
    ACKNOWLEDGMENT_TEXT,
    REFERENCES_LIST
)

def build_paper():
    base_dir = os.path.abspath(r"c:\Users\HP\Desktop\MCA Project\MCA Project AI Breed")
    template_path = os.path.abspath(r"C:\Users\HP\Downloads\conference-template-a4.docx")
    out_docx = os.path.join(base_dir, "IEEE_Project_Paper.docx")
    out_pdf = os.path.join(base_dir, "IEEE_Project_Paper.pdf")

    print(f"Loading IEEE Template from: {template_path}")
    print(f"Target DOCX: {out_docx}")
    print(f"Target PDF:  {out_pdf}")

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    doc = word.Documents.Open(template_path)

    # -------------------------------------------------------------
    # 1. Section 1: Paper Title
    # -------------------------------------------------------------
    sec1 = doc.Sections(1)
    r1 = sec1.Range
    r1.End = r1.End - 1
    r1.Text = PAPER_METADATA["title"] + "\r"
    r1.Paragraphs(1).Range.Style = doc.Styles("paper title")

    # -------------------------------------------------------------
    # 2. Section 2: Authors Block (2 Authors in 2 Columns)
    # -------------------------------------------------------------
    sec2 = doc.Sections(2)
    sec2.PageSetup.TextColumns.SetCount(2)
    sec2.PageSetup.TextColumns.Spacing = 36.0  # 0.5 inches
    a1 = PAPER_METADATA["authors"][0]
    a2 = PAPER_METADATA["authors"][1]
    r2 = sec2.Range
    r2.End = r2.End - 1
    r2.Text = (
        f"{a1['name']}\r{a1['dept']}\r{a1['institution']}\r{a1['city_country']}\r{a1['email']}\r"
        f"\x0e"  # Column break
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

    # -------------------------------------------------------------
    # 3. Section 3: Clear redundant template author section
    # -------------------------------------------------------------
    sec3 = doc.Sections(3)
    r3 = sec3.Range
    r3.End = r3.End - 1
    r3.Text = ""

    # -------------------------------------------------------------
    # 4. Section 4: Two-Column Body Content
    # -------------------------------------------------------------
    sec4 = doc.Sections(4)
    r = sec4.Range
    r.End = r.End - 1
    r.Text = ""  # Clear all sample text

    def add_paragraph(text, style_name, sb=0, sa=2):
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
        # Single paragraph for table head so automatic numbering applies once:
        # e.g., 'TABLE I.  DATASET DISTRIBUTION AND PARTITIONS'
        add_paragraph(table_data["title"], "table head", sb=6, sa=3)

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
        p_sp.Range.Font.Size = 3
        r.SetRange(p_sp.Range.End, p_sp.Range.End)

    def insert_figure(img_rel_path, caption_text, width_pt=225):
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

        clean_cap = caption_text
        if clean_cap.startswith("Fig. "):
            parts = clean_cap.split(".  ", 1)
            if len(parts) > 1:
                clean_cap = parts[1]
            else:
                parts2 = clean_cap.split(". ", 1)
                if len(parts2) > 1:
                    clean_cap = parts2[1]

        add_paragraph(clean_cap, "figure caption", sb=2, sa=5)

    def insert_equation(eq_text, eq_num_str):
        eq_line = f"\t{eq_text}\t{eq_num_str}"
        p_eq = add_paragraph(eq_line, "equation", sb=2, sa=2)
        p_eq.Font.Name = "Times New Roman"
        p_eq.Font.Size = 9

    # --- Abstract ---
    p_abs = add_paragraph("Abstract\u2014" + PAPER_METADATA["abstract"], "Abstract", sb=4, sa=3)
    r_lead = doc.Range(p_abs.Start, p_abs.Start + 9)
    r_lead.Bold = True
    r_lead.Italic = True

    # --- Keywords ---
    p_kw = add_paragraph("Keywords\u2014" + PAPER_METADATA["keywords"], "Keywords", sb=2, sa=6)
    r_kw = doc.Range(p_kw.Start, p_kw.Start + 9)
    r_kw.Bold = True
    r_kw.Italic = True

    # --- Section I: Introduction ---
    sec_i = SECTIONS[0]
    add_paragraph(sec_i["title"], "Heading 1", sb=6, sa=2)
    for p in sec_i["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=2)

    # --- Section II: Related Work ---
    sec_ii = SECTIONS[1]
    add_paragraph(sec_ii["title"], "Heading 1", sb=6, sa=2)
    for p in sec_ii["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=2)

    # --- Section III: Dataset and Preprocessing ---
    sec_iii = SECTIONS[2]
    add_paragraph(sec_iii["title"], "Heading 1", sb=6, sa=2)
    add_paragraph(sec_iii["paragraphs"][0], "Body Text", sb=0, sa=2)
    sub_a = sec_iii["paragraphs"][1].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=2)
    add_paragraph(sec_iii["paragraphs"][2], "Body Text", sb=0, sa=2)
    sub_b = sec_iii["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=2)
    sub_c = sec_iii["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=2)
    insert_ieee_table(TABLES_DATA[0])  # Table I
    sub_d = sec_iii["paragraphs"][5].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=2)

    # --- Section IV: Proposed Methodology ---
    sec_iv = SECTIONS[3]
    add_paragraph(sec_iv["title"], "Heading 1", sb=6, sa=2)
    sub_a = sec_iv["paragraphs"][0].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=2)
    add_paragraph(sec_iv["paragraphs"][1], "Body Text", sb=0, sa=2)
    insert_figure(
        "documentation_assets/figures/workflow_diagram.png",
        "Proposed end-to-end intelligent breed classification pipeline.",
        width_pt=220
    )
    sub_b = sec_iv["paragraphs"][2].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=2)
    sub_c = sec_iv["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=2)
    insert_equation("L_{LS}(y, \\hat{y}) = -(1 - \\epsilon)\\sum_{k=1}^K y_k \\log \\hat{y}_k - \\frac{\\epsilon}{K}\\sum_{k=1}^K \\log \\hat{y}_k", "(1)")
    sub_d = sec_iv["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=2)
    insert_equation("L_{Grad-CAM}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)", "(2)")

    # --- Section V: System Implementation ---
    sec_v = SECTIONS[4]
    add_paragraph(sec_v["title"], "Heading 1", sb=6, sa=2)
    for p in sec_v["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=2)

    # --- Section VI: Experimental Results ---
    sec_vi = SECTIONS[5]
    add_paragraph(sec_vi["title"], "Heading 1", sb=6, sa=2)
    sub_a = sec_vi["paragraphs"][0].split("\n", 1)
    add_paragraph(sub_a[0].replace("A. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_a[1], "Body Text", sb=0, sa=2)
    sub_b = sec_vi["paragraphs"][1].split("\n", 1)
    add_paragraph(sub_b[0].replace("B. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_b[1], "Body Text", sb=0, sa=2)
    insert_ieee_table(TABLES_DATA[1])  # Table II
    sub_c = sec_vi["paragraphs"][2].split("\n", 1)
    add_paragraph(sub_c[0].replace("C. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_c[1], "Body Text", sb=0, sa=2)
    insert_ieee_table(TABLES_DATA[2])  # Table III
    sub_d = sec_vi["paragraphs"][3].split("\n", 1)
    add_paragraph(sub_d[0].replace("D. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_d[1], "Body Text", sb=0, sa=2)
    insert_figure(
        "documentation_assets/charts/v2_training_curves.png",
        "Training and validation loss and accuracy convergence trajectories across 50 epochs.",
        width_pt=220
    )
    insert_figure(
        "documentation_assets/charts/confusion_matrix.png",
        "Normalized confusion matrix across 82 indigenous Indian livestock classes on the locked test partition.",
        width_pt=210
    )
    sub_e = sec_vi["paragraphs"][4].split("\n", 1)
    add_paragraph(sub_e[0].replace("E. ", ""), "Heading 2", sb=4, sa=1)
    add_paragraph(sub_e[1], "Body Text", sb=0, sa=2)
    insert_figure(
        "documentation_assets/figures/fig4_gradcam_composite.png",
        "Grad-CAM visual interpretability overlays for Gir cattle (top) and Bhadawari buffalo (bottom).",
        width_pt=220
    )

    # --- Section VII: Discussion ---
    sec_vii = SECTIONS[6]
    add_paragraph(sec_vii["title"], "Heading 1", sb=6, sa=2)
    for p in sec_vii["paragraphs"]:
        if any(p.startswith(f"{c}. ") for c in ["A", "B", "C"]):
            lines = p.split("\n", 1)
            add_paragraph(lines[0][3:], "Heading 2", sb=4, sa=1)
            if len(lines) > 1:
                add_paragraph(lines[1], "Body Text", sb=0, sa=2)
        else:
            add_paragraph(p, "Body Text", sb=0, sa=2)

    # --- Section VIII: Conclusion and Future Work ---
    sec_viii = SECTIONS[7]
    add_paragraph(sec_viii["title"], "Heading 1", sb=6, sa=2)
    for p in sec_viii["paragraphs"]:
        add_paragraph(p, "Body Text", sb=0, sa=2)

    # --- Acknowledgment ---
    add_paragraph("ACKNOWLEDGMENT", "Heading 5", sb=6, sa=2)
    add_paragraph(ACKNOWLEDGMENT_TEXT, "Body Text", sb=0, sa=2)

    # --- References ---
    add_paragraph("REFERENCES", "Heading 5", sb=6, sa=2)
    for ref_text in REFERENCES_LIST:
        p_r = add_paragraph(ref_text, "references", sb=0, sa=1)
        p_r.Font.Name = "Times New Roman"
        p_r.Font.Size = 8

    # --- Clean up any template leftover shapes or footers ---
    while doc.Shapes.Count > 0:
        doc.Shapes(1).Delete()

    for s in doc.Sections:
        for f_idx in [1, 2, 3]:
            try:
                s.Footers(f_idx).Range.Text = ""
            except:
                pass
        for h_idx in [1, 2, 3]:
            try:
                s.Headers(h_idx).Range.Text = ""
            except:
                pass

    doc.SaveAs2(out_docx, 16)
    doc.SaveAs2(out_pdf, 17)

    pages = doc.ComputeStatistics(2)
    print(f"Final IEEE Paper Generated: {pages} Pages.")
    doc.Close(False)
    word.Quit()
    return True

if __name__ == "__main__":
    build_paper()
