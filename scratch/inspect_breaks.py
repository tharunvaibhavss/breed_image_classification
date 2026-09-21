import docx

doc = docx.Document(r'C:\Users\HP\Desktop\MCA Project\docs\Template 1 MCA - Project I Documentation Template for Research Application Based Project.docx')

leaf_indices = [40, 86, 124, 138, 178, 192, 210, 224, 238, 252, 266, 280, 294, 308]

for i in range(len(leaf_indices) - 1):
    start = leaf_indices[i]
    end = leaf_indices[i+1]
    count = end - start - 1
    print(f'Between {start} and {end}: {count} paragraphs')
    # sample one paragraph's xml
    sample_xml = doc.element.body[start+1].xml
    print(f'   Sample XML: {sample_xml[:150]}...')
