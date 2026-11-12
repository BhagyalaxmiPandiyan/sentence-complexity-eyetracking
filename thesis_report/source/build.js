// Builds the MTP thesis report (.docx) in the IIT Jodhpur Annexure-I format.
const fs = require("fs");
const L = require("./lib.js");
const { d, FONT, LINE, P, H1, runs, REFS, order } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber, NumberFormat,
        TableOfContents, LevelFormat, HeadingLevel } = d;

const INFO = {
  title: "Measuring Sentence Complexity from Eye-Tracking Measures: An English Replication with GPT-2 Surprisal and a Morphology-Aware Extension to Tamil",
  name: "Bhagyalaxmi Pandiyan", roll: "M25DE2070", supervisor: "Dr. Sidharth Ranjan",
  degree: "Master of Technology", programme: "Data Engineering", dept: "Data Engineering",
  date: "November 2026",
};

const T = (text, o = {}) => new TextRun({ text, font: FONT, ...o });
const line = (children, o = {}) => new Paragraph({ children: Array.isArray(children) ? children : [children], ...o });
const right = (children, after = 0) => line(children, { alignment: AlignmentType.RIGHT, spacing: { after } });
const gap = n => line(T(""), { spacing: { after: n } });

// ── cover (counted, not numbered)
const cover = [
  gap(1400),
  right(T(INFO.title, { bold: true, size: 36 }), 1600),
  right(T("A Project Report submitted by", { italics: true, size: 26 }), 80),
  right(T(INFO.name, { bold: true, size: 30 }), 1000),
  right(T("in partial fulfillment of the requirements for the award of the degree of", { italics: true, size: 26 }), 80),
  right(T(`${INFO.degree} in ${INFO.programme}`, { bold: true, size: 30 }), 2600),
  right(T("[Insert IIT Jodhpur logo here]", { color: "808080", size: 20 }), 600),
  right(T("Indian Institute of Technology Jodhpur", { bold: true, size: 26 }), 40),
  right(T(INFO.dept, { bold: true, size: 26 }), 40),
  right(T(INFO.date, { italics: true, size: 26 })),
];

// ── front matter (lower-case Roman numbers)
const just = (children, o = {}) => new Paragraph({ children, alignment: AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 160 }, ...o });
const frontTitle = (t, pb = true) => new Paragraph({ alignment: AlignmentType.RIGHT, pageBreakBefore: pb, spacing: { after: 300 },
  children: [T(t, { bold: true, size: 28 })] });

const declaration = [
  frontTitle("Declaration", false),
  just([T("I hereby declare that the work presented in this Project Report titled "), T(INFO.title, { italics: true }),
    T(` submitted to the Indian Institute of Technology Jodhpur in partial fulfilment of the requirements for the award of the degree of ${INFO.degree} in ${INFO.programme}, is a bonafide record of the research work carried out under the supervision of ${INFO.supervisor}. The contents of this Project Report in full or in parts, have not been submitted to, and will not be submitted by me to, any other Institute or University in India or abroad for the award of any degree or diploma.`)]),
  gap(500),
  right(T("Signature", { bold: true })), right(T(INFO.name, { italics: true })), right(T(INFO.roll)),
];
const certificate = [
  frontTitle("Certificate"),
  just([T("This is to certify that the Project Report titled "), T(INFO.title, { italics: true }),
    T(`, submitted by ${INFO.name} (${INFO.roll}) to the Indian Institute of Technology Jodhpur for the award of the degree of ${INFO.degree} in ${INFO.programme}, is a bonafide record of the research work done under my supervision. To the best of my knowledge, the contents of this report, in full or in parts, have not been submitted to any other Institute or University for the award of any degree or diploma.`)]),
  gap(700),
  right(T("Signature", { bold: true })), right(T(INFO.supervisor)),
];
const acknowledgements = [
  frontTitle("Acknowledgements"),
  just([T(`I am grateful to my supervisor, ${INFO.supervisor}, for the guidance and feedback given throughout this project. I thank the Tamil readers who gave their time to rate sentences and to take part in the eye-tracking study, and the creators of the Dundee corpus, StoryWeaver, UD Tamil-TTB and the open-source tools on which this work is built.`)]),
  just([T("Use of AI tools. ", { bold: true }), T("Large language models were part of this work in two ways. As research objects, Claude Opus 5.5 produced the Tamil simplifications that were evaluated, and Gemini was evaluated as an LLM judge and is used for live simplification in the toolkit; both uses are reported in the text. As assistants, AI tools were used to help write and debug code and to help draft and edit this report. All experiments were run with the project code, all reported numbers come from its output files, and I have checked the content of this report.")]),
];
const ABSTRACT = "This thesis studies how sentence complexity can be measured from human reading behaviour, and how such measures transfer from English to Tamil. First, the two-stage model of Singh et al. is replicated on the Dundee eye-tracking corpus: System 1 predicts four word-level reading-time measures from linguistic features, and System 2 uses these predictions to decide which sentence of a Wikipedia–Simple Wikipedia pair is harder. System 2 reaches 74.15% and 75.26% accuracy, matching the original study. Adding contextual GPT-2 surprisal improves all four reading-time models and exceeds the original results for regression-path and total duration. Second, word-level and morphology-aware difficulty metrics are built for Tamil and tested on graded texts, human-written simplifications, a rating study of LLM simplifications, and a webcam eye-tracking study. Morphological features carry signal on their own but add no reliable gain over word-level features. All components are available in an interactive toolkit.";
const nWords = ABSTRACT.split(/\s+/).length;
if (nWords > 150) throw new Error(`abstract has ${nWords} words`);
const abstract = [frontTitle("Abstract"), just([T(ABSTRACT)], { indent: { firstLine: 720 } })];

const tocTitle = t => frontTitle(t);
const contents = [
  tocTitle("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  tocTitle("List of Figures"),
  new TableOfContents("Figures", { hyperlink: true, stylesWithLevels: [{ styleName: "FigureCaption", level: 1 }] }),
  tocTitle("List of Tables"),
  new TableOfContents("Tables", { hyperlink: true, stylesWithLevels: [{ styleName: "TableCaption", level: 1 }] }),
];

// ── main text, then references (order of first citation), then appendices
const main = require("./content.js")();
const refs = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [T("REFERENCES")] }),
  ...order.map((k, i) => new Paragraph({
    children: [T(`[${i + 1}]`), new TextRun({ text: "\t", font: FONT }), ...runs(REFS[k])],
    tabStops: [{ type: "left", position: 600 }], indent: { left: 600, hanging: 600 },
    alignment: AlignmentType.LEFT, spacing: { line: 276, after: 100 } })),
];
const appendix = require("./content.js").appendix();
const unused = Object.keys(REFS).filter(k => !order.includes(k));
if (unused.length) console.warn("uncited references:", unused.join(", "));

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] });
const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1080, right: 1080, footer: 600 } };

const doc = new Document({
  creator: INFO.name, title: INFO.title, description: "M.Tech. project report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, allCaps: true },
        paragraph: { spacing: { before: 0, after: 360 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, italics: true, bold: false },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, italics: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
      { id: "FigureCaption", name: "FigureCaption", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 22, italics: true } },
      { id: "TableCaption", name: "TableCaption", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 22, italics: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "obj", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } }, children: cover },
    { properties: { page: { ...page, pageNumbers: { start: 2, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer }, children: [...declaration, ...certificate, ...acknowledgements, ...abstract, ...contents] },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: footer }, children: [...main, ...refs, ...appendix] },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(process.argv[2], buf);
  console.log(`wrote ${process.argv[2]} | abstract ${nWords} words | ${order.length} references`);
});
