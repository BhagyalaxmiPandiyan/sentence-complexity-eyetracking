/**
 * Receives the Tamil reading study data and writes it into this Google Sheet,
 * one upload per sentence as soon as the reader has finished it.
 *
 * Setup (once): Google Sheet -> Extensions -> Apps Script -> paste this file ->
 * Deploy -> New deployment -> type "Web app" -> Execute as: Me,
 * Who has access: Anyone -> Deploy. Copy the web-app URL into config.js (ENDPOINT).
 *
 * Tabs (created automatically):
 *   participants : one row per reader (age, gender, sentence order, screen, browser)
 *   validation   : gaze accuracy at start, each break and the end (median error, px)
 *   sentences    : one row per sentence: reading time (sentence on -> ENTER), gaze
 *                  quality, noise filter result, question, answer, correct,
 *                  word boxes and the raw gaze (for re-analysis)
 *   words        : one row per word, Dundee columns + FFD, FPD, RPD, TD (ms)
 *   fixations    : one row per event, Dundee RT_dataset1P (MA1P) columns
 */
const COLUMNS = {
  participants: ["ts", "pid", "version", "type", "age", "gender", "native_tamil", "glasses", "order",
    "gaze_source", "screen_w", "screen_h", "window_w", "window_h", "dpr", "user_agent", "sentences"],
  validation: ["ts", "pid", "version", "label", "points_with_data", "median_error_px",
    "median_error_x_px", "median_error_y_px", "window_w", "window_h"],
  sentences: ["ts", "pid", "version", "trial_no", "practice", "sid", "band", "screen", "n_words", "n_lines",
    "font_px", "reading_time_ms", "samples", "valid_gaze_pct", "fixations", "too_long_dropped",
    "off_text_pct", "sentence_ok", "excluded_because", "drift_x", "drift_y", "probe_word",
    "probe_answer", "response", "correct", "answer_rt_ms", "window_w", "window_h", "boxes", "raw_gaze"],
  words: ["ts", "pid", "trial_no", "sid", "band", "TEXT", "WNUM", "WORD", "LINE", "LINE_POS", "SERIAL",
    "START_POS", "OLEN", "WLEN", "PUNCT", "TXFR", "FFD", "FPD", "RPD", "TD", "N_FIX", "FIXATED",
    "SENTENCE_OK"],
  fixations: ["ts", "pid", "trial_no", "sid", "FXNO", "WORD", "TEXT", "LINE", "OLEN", "WLEN", "XPOS",
    "WNUM", "FDUR", "OBLP", "WDLP", "LAUN", "TXFR"],
};
const CELL_MAX = 49000; // Google Sheets limit is 50,000 characters per cell

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const d = JSON.parse(e.postData.contents);
    const common = { ts: d.ts, pid: d.pid, version: d.version };
    if (d.type === "participant" || d.type === "participant_end") {
      append_("participants", [Object.assign({}, d)]);
    } else if (d.type === "validation") {
      append_("validation", [d]);
    } else if (d.type === "sentence") {
      append_("sentences", [Object.assign({}, common, d.sentence)]);
      if (d.words) append_("words", d.words.map(r => Object.assign({}, common, r)));
      if (d.fixations) append_("fixations", d.fixations.map(r => Object.assign({}, common, r)));
    }
    return ContentService.createTextOutput("ok");
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return ContentService.createTextOutput("Tamil reading study receiver is running.");
}

function append_(name, rows) {
  if (!rows.length) return;
  const cols = COLUMNS[name];
  const sheet = getSheet_(name, cols);
  const values = rows.map(r => cols.map(c => {
    const v = r[c] === undefined || r[c] === null ? "" : r[c];
    if (typeof v === "string" && v.length > CELL_MAX) return v.slice(0, CELL_MAX) + "…TRUNCATED";
    return v;
  }));
  sheet.getRange(sheet.getLastRow() + 1, 1, values.length, cols.length).setValues(values);
}

function getSheet_(name, cols) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(cols);
    sh.setFrozenRows(1);
    sh.getRange(1, 1, 1, cols.length).setFontWeight("bold");
    // Words and ids as plain text, so Sheets does not turn them into dates or numbers
    ["pid", "sid", "WORD", "boxes", "raw_gaze", "order", "probe_word", "excluded_because"].forEach(c => {
      const k = cols.indexOf(c);
      if (k >= 0) sh.getRange(1, k + 1, sh.getMaxRows(), 1).setNumberFormat("@");
    });
  }
  return sh;
}
