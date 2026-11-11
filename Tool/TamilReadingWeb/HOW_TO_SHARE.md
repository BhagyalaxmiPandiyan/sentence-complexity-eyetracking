# Tamil Reading Study (web) — how to set up and share

Participants open one link in **Chrome or Edge on a laptop/desktop with a webcam**.
Nothing to install. Each sentence stays on screen until they press **ENTER**; right
then the app computes FFD / FPD / RPD / TD for every word of that sentence and sends
that sentence's rows to your Google Sheet.

## Files

| File | What it is |
|---|---|
| `index.html` | the page (loads WebGazer 3.5.3 for webcam gaze) |
| `app.js` | the study: form → camera → calibration → 3 practice → 100 sentences in 4 parts |
| `measures.js` | fixations + FFD/FPD/RPD/TD + noise filter (same as the desktop tool) |
| `stimuli.js` | the 100 sentences (easy/medium/hard) + 3 practice + 20 questions |
| `config.js` | settings — **put your Google Sheet link (ENDPOINT) here** |
| `apps_script.gs` | receiver code to paste into the Google Sheet |

## Step 1 — test on your own computer (no Google Sheet yet)

```
cd "E:\Project\Final One\Tool\TamilReadingWeb"
python -m http.server 8000
```
Open http://localhost:8000 in Chrome. With `ENDPOINT` empty this is **test mode**:
nothing is uploaded; at the end your data downloads as a `.json` file.
`http://localhost:8000/?gaze=mouse` uses the mouse as "gaze" (for checking the flow only).

## Step 2 — the Google Sheet (once)

1. Create a new Google Sheet, e.g. *Tamil Reading Study data*.
2. **Extensions → Apps Script**. Delete what is there, paste all of `apps_script.gs`, **Save**.
3. **Deploy → New deployment → ⚙ Select type: Web app**
   - Execute as: **Me**
   - Who has access: **Anyone**
   - **Deploy** → allow access with your Google account.
4. Copy the **Web app URL** (ends in `/exec`). Open it in a browser — it should say
   *"Tamil reading study receiver is running."*
5. Paste it in `config.js`:  `ENDPOINT: "https://script.google.com/macros/s/.../exec",`

The tabs fill in by themselves:

| Tab | One row per | Columns |
|---|---|---|
| `participants` | reader | age, gender, Tamil first language, glasses, sentence order, screen, browser |
| `validation` | accuracy check | median gaze error in px (start, each break, end) |
| `sentences` | sentence read | reading time (sentence → ENTER), valid gaze %, fixations, noise-filter result, question, answer, correct, word boxes, raw gaze |
| `words` | word | Dundee columns (TEXT, WNUM, WORD, LINE, OLEN, WLEN, …) + **FFD, FPD, RPD, TD**, FIXATED, SENTENCE_OK |
| `fixations` | fixation / blink | Dundee RT_dataset1P columns (FXNO, WORD, TEXT, LINE, XPOS, WNUM, FDUR, OBLP, WDLP, LAUN, TXFR) |

**File → Download → Microsoft Excel (.xlsx)** gives the whole study as one Excel file.

## Step 3 — put it online

The page must be on **https** (browsers only allow the webcam there). Free options:

- **GitHub Pages**: put the `TamilReadingWeb` folder in a GitHub repository →
  Settings → Pages → deploy from branch. Link looks like
  `https://<user>.github.io/<repo>/Tool/TamilReadingWeb/`
- **Netlify Drop** (app.netlify.com/drop): drag the folder in, get a link.

## Step 4 — check the Google Sheet connection (before sharing)

1. Open `https://…/TamilReadingWeb/?check=1`. It sends one test row.
2. In your Google Sheet, the **validation** tab should now have a row with pid **CHECK**
   and label **connection_test**. If yes, saving works; delete that row.
3. Then do the study once yourself (`?pid=PILOT`) and watch rows appear in
   `sentences`, `words` and `fixations` after each sentence.

If no row appears: the Apps Script must be deployed as **Web app** with access **Anyone**,
and `config.js` must have the URL ending in **/exec**. After changing the script, use
**Deploy → Manage deployments → Edit → New version** (the URL stays the same).

The page refuses to start on the internet while `ENDPOINT` is empty, so nobody can read
100 sentences into a page that saves nothing.

## Step 5 — share

Give each participant their own link with their code:
`https://…/TamilReadingWeb/?pid=R01`

The site shows the full instructions (what will happen, the keys, a ready-checklist)
before the camera starts, and shows progress (part 1/4 · 12/100) during reading.

## Noise filter (same as the desktop tool)

1. Gaze smoothed (running median of 3) + WebGazer's Kalman filter
2. Fixations < 80 ms not counted, > 1200 ms dropped as noise
3. Sentence excluded (`SENTENCE_OK = 0`) if valid gaze < 70 % or > 50 % of fixation
   time is off the text
4. Outliers (above mean + 3 SD per reader) need all of a reader's sentences, so they
   are marked afterwards, when the data is analysed

## Good to know

- Webcam gaze in a browser is much less precise than a lab eye tracker (typically
  100–200 px). The app uses large text, wide word gaps and 3× line spacing, checks
  accuracy after calibration (asks to recalibrate above 150 px), recalibrates at every
  break, and corrects drift on the dot before every sentence.
- No video or images are stored or sent — only screen coordinates.
- If the page is reloaded, the reader continues where they stopped (after a quick
  recalibration). Data that could not be sent (no internet) is kept and retried.
