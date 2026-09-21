import pypdf

reader = pypdf.PdfReader(r'c:\Users\HP\Desktop\MCA Project\MCA Project AI Breed\MCA_Project_Documentation.pdf')

targets = [
    'CHAPTER 1: INTRODUCTION',
    '1.1 Introduction to the Project',
    '1.2 Problem Statement',
    '1.3 Objectives of the Study',
    '1.4 Scope of the Project',
    '1.5 Organization of the Report',
    'CHAPTER 2: LITERATURE REVIEW',
    '2.1 Review of Relevant',
    '2.2 Summary of Literature',
    '2.3 Research Gap',
    'CHAPTER 3: SYSTEM REQUIREMENTS',
    '3.1 Software Requirements',
    '3.2 Hardware Requirements',
    '3.3 Tech Stack',
    'CHAPTER 4: PROPOSED METHODOLOGY',
    '4.1 Proposed Methodology',
    '4.2 Modules and Description',
    '4.3 System Architecture',
    '4.4 Work Flow Diagram',
    '4.5 Database Design',
    '4.6 Input Design',
    '4.7 Output Design',
    'CHAPTER 5: SYSTEM IMPLEMENTATION',
    '5.1 Algorithm Implementation',
    '5.2 Coding and Development',
    '5.3 Implementation Tools',
    'CHAPTER 6: RESULTS AND DISCUSSIONS',
    '6.1 Performance Metrics',
    '6.2 Implementation Results',
    '6.3 Discussions',
    'CHAPTER 7: CONCLUSION',
    '7.1 Conclusion',
    '7.2 Scope for Future',
    'BIBLIOGRAPHY / REFERENCES',
    'APPENDIX A: KEY SOURCE CODE',
    'APPENDIX B: OUTPUT SCREENS',
    'RESEARCH PAPER PUBLICATION STATUS'
]

found = {}
# Skip first 15 pages (Title, certificates, TOC)
for i in range(15, len(reader.pages)):
    txt = reader.pages[i].extract_text()
    for t in targets:
        if t not in found and t.lower() in txt.lower():
            found[t] = i + 1

for t in targets:
    p_num = found.get(t, 'Not Found')
    print(f'{t:35} : Page {p_num}')
