import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

template_path = r'C:\Users\HP\Desktop\MCA Project\docs\Template 1 MCA - Project I Documentation Template for Research Application Based Project.docx'
doc = docx.Document(template_path)

# Let's find child 210 (Chapter 1) and child 224 (Chapter 2)
c1_tbl = doc.element.body[210]
c2_tbl = doc.element.body[224]

# Insert a test paragraph right before c2_tbl
p = doc.add_paragraph()
p.text = "This is a test paragraph in Chapter 1."
p_elem = p._element
# Move p_elem right before c2_tbl
c2_tbl.addprevious(p_elem)

print("Successfully inserted paragraph before Chapter 2 leaf table!")
