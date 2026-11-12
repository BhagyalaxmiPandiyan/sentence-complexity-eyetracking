# Thesis report source

The report is generated, not typed: `content.js` holds the chapter text, `refs.js` the IEEE
references (numbered automatically in order of first citation), `lib.js` the IIT Jodhpur
formatting (A4, margins 1/1/0.75/0.75 in, Times New Roman 12, 1.5 spacing, justified), and
`build.js` the cover, declaration, certificate, abstract, contents lists and page numbering.

```
npm install docx@9.5.1
python make_figs.py fig          # figures (fig5_1 and fig6_1 are copied/screenshotted)
node build.js thesis_raw.docx
powershell -File finalize.ps1 -In <abs>\thesis_raw.docx -OutDocx <abs>\thesis.docx -OutPdf <abs>\thesis.pdf
```
`finalize.ps1` opens the file in Microsoft Word to fill in the table of contents and export the PDF.
Small text edits can also be made directly in the .docx (then right-click the contents → Update Field).
