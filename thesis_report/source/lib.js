// Formatting helpers for the IIT Jodhpur project report (Annexure I).
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, WidthType,
  BorderStyle, ShadingType, TabStopType, Tab, HeadingLevel, VerticalAlign,
} = d;

const FONT = "Times New Roman";
const TEXT_W = 11906 - 2 * 1080;          // A4 width minus 0.75 in left/right = 9746 DXA
const LINE = 360;                          // 1.5 line spacing

// ── citations: [@key] or [@a; @b] -> IEEE numbers in order of first citation
const REFS = require("./refs.js");
const order = [];
function cite(keys) {
  return keys.map(k => {
    if (!REFS[k]) throw new Error("unknown reference " + k);
    if (!order.includes(k)) order.push(k);
    return `[${order.indexOf(k) + 1}]`;
  }).join(", ");
}

// ── inline markup: **bold**, *italic*, _{sub}, ^{sup}, [@cite]
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|\[@[^\]]+\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: cite(t.slice(1, -1).split(";").map(s => s.trim().replace(/^@/, ""))), font: FONT, ...base }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { line: LINE, after: 120 }, indent: opts.noIndent ? undefined : { firstLine: 0 }, ...opts,
});
const H1 = (t, pageBreak = true) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: pageBreak, children: [new TextRun({ text: t.toUpperCase(), font: FONT })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const bullets = items => items.map(t => new Paragraph({
  children: runs(t), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { line: LINE, after: 60 },
}));
const numbered = (items, ref = "numbers") => items.map(t => new Paragraph({
  children: runs(t), numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { line: LINE, after: 60 },
}));

// Equation: centred, number in parentheses flush right
const EQ = (eq, num) => new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
  spacing: { before: 120, after: 120, line: LINE },
  children: [new TextRun({ children: [new Tab()], font: FONT }), ...runs(eq, { italics: false }),
             new TextRun({ children: [new Tab(), `(${num})`], font: FONT })],
});

// Table: caption above (centred), single-spaced 10.5 pt cells
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function TABLE(caption, header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = TEXT_W / total;
  const w = widths.map(x => Math.round(x * scale));
  w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { line: 240, before: 0, after: 0 }, children: runs(String(t), head ? { size: 21, bold: true } : { size: 21 }) })],
  });
  return [
    new Paragraph({ style: "TableCaption", alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 200, after: 100, line: 276 }, children: runs(caption) }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w, alignment: AlignmentType.CENTER,
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
             ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// Figure: centred image, caption below
function FIG(file, caption, widthIn) {
  const buf = fs.readFileSync(file);
  const { w, h } = pngSize(buf);
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ style: "FigureCaption", alignment: AlignmentType.CENTER, spacing: { after: 200, line: 276 },
      children: runs(caption) }),
  ];
}
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

module.exports = { d, FONT, TEXT_W, LINE, runs, P, H1, H2, H3, bullets, numbered, EQ, TABLE, FIG, REFS, order };
