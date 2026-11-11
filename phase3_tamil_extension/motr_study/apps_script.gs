/**
 * Receives MoTR study data and writes it into this Google Sheet.
 * Setup (once): Google Sheet -> Extensions -> Apps Script -> paste this file -> Deploy ->
 * New deployment -> type "Web app" -> Execute as: Me, Who has access: Anyone -> Deploy.
 * Copy the web-app URL into web/config.js (ENDPOINT).
 *
 * Two tabs are created automatically:
 *   participants : one row per reader (age, gender, education, sentence order, device)
 *   trials       : one row per sentence read (word boxes, mouse visits, raw mouse path, probe answer)
 */
const P_COLS = ["ts", "pid", "version", "age", "gender", "native_tamil", "education", "news_reading",
  "order", "screen_w", "screen_h", "window_w", "window_h", "dpr", "user_agent"];
const T_COLS = ["ts", "pid", "version", "practice", "trial_no", "sid", "text", "n_words", "rt_ms",
  "aois", "visits", "raw", "stage_w", "stage_h", "resized",
  "probe_word", "probe_answer", "response", "correct", "probe_rt_ms"];
const CELL_MAX = 49000; // Google Sheets limit is 50,000 characters per cell

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const d = JSON.parse(e.postData.contents);
    const isP = d.type === "participant";
    const cols = isP ? P_COLS : T_COLS;
    const sheet = getSheet_(isP ? "participants" : "trials", cols);
    sheet.appendRow(cols.map(c => {
      const v = d[c] === undefined ? "" : String(d[c]);
      return v.length > CELL_MAX ? v.slice(0, CELL_MAX) + "…TRUNCATED" : v;
    }));
    return ContentService.createTextOutput("ok");
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return ContentService.createTextOutput("MoTR Tamil receiver is running.");
}

function getSheet_(name, cols) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) {
    sh = ss.insertSheet(name);
    sh.appendRow(cols);
    sh.setFrozenRows(1);
    sh.getRange("A:Z").setNumberFormat("@"); // keep everything as plain text (no auto date/number guessing)
  }
  return sh;
}
