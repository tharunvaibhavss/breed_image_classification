import os
from pathlib import Path
import docx
import win32com.client

ws = Path(r'c:\Users\HP\Desktop\MCA Project\MCA Project AI Breed')
template_path = r'C:\Users\HP\Desktop\MCA Project\docs\Template 1 MCA - Project I Documentation Template for Research Application Based Project.docx'
test_docx = ws / 'scratch' / 'test_doc.docx'
test_pdf = ws / 'scratch' / 'test_doc.pdf'

doc = docx.Document(template_path)
doc.save(str(test_docx))
print("Saved test_doc.docx")

# Convert using Word COM
word = win32com.client.Dispatch("Word.Application")
word.Visible = False
try:
    doc_com = word.Documents.Open(str(test_docx.resolve()))
    # 17 corresponds to wdFormatPDF
    doc_com.SaveAs(str(test_pdf.resolve()), FileFormat=17)
    doc_com.Close()
    print(f"Successfully converted to PDF: {test_pdf} (size: {test_pdf.stat().st_size} bytes)")
finally:
    word.Quit()
