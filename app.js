// Tamil reading study in the browser with webcam eye tracking (WebGazer).
// Each sentence stays on screen until the reader presses ENTER. Right then
// the gaze recorded for that sentence is turned into fixations and
// FFD / FPD / RPD / TD per word (measures.js), the optional question is
// asked, and the rows for that one sentence are sent to the Google Sheet.
// Researcher test without a webcam: add ?gaze=mouse to the address.
"use strict";

const VERSION = "tamil-webgaze-1.0";
const STORE_KEY = "tamil_webgaze_state";
const $app = document.getElementById("app");
const MOUSE = new URLSearchParams(location.search).get("gaze") === "mouse";

// ---------- small helpers ----------
const load = k => { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage full or blocked */ } };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = html => { $app.innerHTML = html; window.scrollTo(0, 0); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const now = () => performance.now();
const median = v => { const s = v.slice().sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

function hashStr(s) {
  let h = 2166136261;
  for (const c of s) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function mulberry32(a) {
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Same participant code -> same order (a reload resumes correctly)
function shuffled(arr, seed) {
  const rnd = mulberry32(seed), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

// Wait for one of the keys; returns {key, t}. Ignores auto-repeat.
function waitKey(keys, minMs = 0) {
  const t0 = now();
  return new Promise(resolve => {
    function onKey(e) {
      if (e.repeat || !keys.includes(e.key) || now() - t0 < minMs) return;
      e.preventDefault();
      document.removeEventListener("keydown", onKey);
      resolve({ key: e.key, t: now() });
    }
    document.addEventListener("keydown", onKey);
  });
}

// ---------- sentences: Dundee screen and word numbers (sid order) ----------
const BY_SID = Object.fromEntries(STIMULI.map(s => [s.sid, s]));
if (PRACTICE[1] && !PRACTICE[1].probe) { // one demo question so the first real one is no surprise
  const longest = PRACTICE[1].text.split(/\s+/).reduce((a, b) => (b.length > a.length ? b : a)).replace(/[.,]/g, "");
  PRACTICE[1].probe = { word: longest, answer: "yes" };
}
const SCREEN = {}, FIRST_WNUM = {}, FREQ = new Map();
{
  let wnum = 1;
  STIMULI.slice().sort((a, b) => a.sid.localeCompare(b.sid)).forEach((s, i) => {
    SCREEN[s.sid] = i + 1;
    FIRST_WNUM[s.sid] = wnum;
    const words = s.text.split(/\s+/);
    wnum += words.length;
    for (const w of words) { const b = Measures.splitMarks(w).bare; FREQ.set(b, (FREQ.get(b) || 0) + 1); }
  });
}

// ---------- state + upload queue (survives a page reload) ----------
let state = null; // {pid, info, order, next, practiceDone, fxno, pending, results}
const persist = () => save(STORE_KEY, state);

let flushing = false;
async function flush() {
  if (!CONFIG.ENDPOINT || flushing) return;
  flushing = true;
  try {
    while (state.pending.length) {
      try {
        await fetch(CONFIG.ENDPOINT, {
          method: "POST", mode: "no-cors",
          headers: { "Content-Type": "text/plain;charset=utf-8" },
          body: JSON.stringify(state.pending[0]),
        });
      } catch (e) {
        setTimeout(flush, 5000); // offline: retry later, nothing is lost
        return;
      }
      state.pending.shift();
      persist();
    }
  } finally {
    flushing = false;
  }
}
function record(payload) {
  payload = { version: VERSION, pid: state.pid, ts: new Date().toISOString(), ...payload };
  if (CONFIG.ENDPOINT) state.pending.push(payload);
  else state.results.push(payload); // test mode: kept for the download
  persist();
  flush();
}
function downloadData() {
  const blob = new Blob([JSON.stringify({ pid: state.pid, info: state.info, results: state.results })],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `tamil_reading_${state.pid}.json`;
  a.click();
}

// ---------- gaze source ----------
let recent = [];     // last 2 s of samples, for drift correction and validation
let sink = null;     // samples of the sentence being read
let offset = { x: 0, y: 0 };

function addSample(x, y, valid) {
  const s = { t: now(), x, y, valid };
  recent.push(s);
  while (recent.length && s.t - recent[0].t > 2000) recent.shift();
  if (sink) sink.push({ t: s.t, x: x - offset.x, y: y - offset.y, valid });
}

async function startGaze() {
  if (MOUSE) {
    let mx = NaN, my = NaN;
    document.addEventListener("mousemove", e => { mx = e.clientX; my = e.clientY; });
    setInterval(() => addSample(mx, my, Number.isFinite(mx)), 33);
    return;
  }
  webgazer.params.showVideoPreview = true;
  webgazer.setRegression("ridge")
    .saveDataAcrossSessions(false)
    .applyKalmanFilter(true)
    .setGazeListener(data => {
      if (data) addSample(data.x, data.y, true);
      else addSample(NaN, NaN, false);
    });
  await webgazer.begin();
  webgazer.removeMouseEventListeners(); // learn only from calibration clicks
  webgazer.showPredictionPoints(false);
}
function showCamera(on) {
  if (MOUSE) return;
  webgazer.showVideoPreview(on);
  webgazer.showFaceOverlay(on);
  webgazer.showFaceFeedbackBox(on);
}

// ---------- screens ----------
function deviceOK() {
  return window.matchMedia("(pointer: fine)").matches && window.innerWidth >= 1000;
}

// Researcher check (?check=1): send one row to the validation tab
async function screenConnectionCheck() {
  if (!CONFIG.ENDPOINT) {
    show(`<div class="card"><h1>Connection check</h1><p class="banner">config.js ENDPOINT is empty —
      paste the Apps Script web-app URL there first.</p></div>`);
    return;
  }
  const stamp = new Date().toISOString();
  try {
    await fetch(CONFIG.ENDPOINT, {
      method: "POST", mode: "no-cors", headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify({ type: "validation", version: VERSION, pid: "CHECK", ts: stamp, label: "connection_test" }),
    });
    show(`<div class="card"><h1>Connection check sent</h1>
      <p>Open your Google Sheet: within a few seconds the <b>validation</b> tab should have a row with
      pid <b>CHECK</b>, label <b>connection_test</b> and time <b>${esc(stamp)}</b>.</p>
      <p>If it is there, the study saves data correctly — you can delete that row and share the link.
      If not: check that the Apps Script was deployed as a <b>Web app</b> with access <b>Anyone</b>, and that
      config.js has the URL ending in <b>/exec</b>.</p></div>`);
  } catch (e) {
    show(`<div class="card"><h1>Connection check failed</h1><p class="banner">The request could not be sent
      (${esc(e.message)}). Check the internet connection and the ENDPOINT URL in config.js.</p></div>`);
  }
}

function screenWelcome() {
  if (new URLSearchParams(location.search).get("check")) { screenConnectionCheck(); return; }
  if (!deviceOK()) {
    show(`<div class="card"><h1>தமிழ் வாசிப்பு ஆய்வு</h1>
      <p>இந்த ஆய்வை <b>வெப்கேமரா உள்ள கணினியில் (லேப்டாப் / டெஸ்க்டாப்)</b> மட்டுமே செய்ய முடியும்.
      உலாவி சாளரம் சிறியதாக இருந்தால், அதைப் பெரிதாக்கி பக்கத்தை மீண்டும் ஏற்றவும்.</p>
      <p class="en">This study needs a laptop or desktop computer with a webcam and a mouse or touchpad,
      and a browser window at least 1000 px wide. Phones and tablets are not supported.</p></div>`);
    return;
  }
  const saved = load(STORE_KEY);
  if (saved && saved.pid && saved.version === VERSION && saved.next < saved.order.length) {
    show(`<div class="card"><h1>மீண்டும் வருக!</h1>
      <p>குறியீடு <b>${esc(saved.pid)}</b> — ${saved.next} / ${saved.order.length} வாக்கியங்கள் முடிந்துள்ளன.
      கேமராவை மீண்டும் அளவீடு செய்த பின், நிறுத்திய இடத்திலிருந்து தொடரலாம்.</p>
      <p class="en">You already started (code ${esc(saved.pid)}, ${saved.next}/${saved.order.length} done).
      After a new calibration you continue where you stopped.</p>
      <button class="btn" id="resume">தொடர்க / Continue</button>
      <button class="btn secondary" id="fresh">புதிதாகத் தொடங்கு / Start new</button></div>`);
    document.getElementById("resume").onclick = () => { state = saved; flush(); screenCamera(); };
    document.getElementById("fresh").onclick = () => {
      if (confirm("முந்தைய முன்னேற்றம் நீக்கப்படும். உறுதியா? / Previous progress will be cleared. Sure?")) {
        localStorage.removeItem(STORE_KEY); screenWelcome();
      }
    };
    return;
  }

  const urlPid = new URLSearchParams(location.search).get("pid") || "";
  // Warn before anyone reads 100 sentences into a page that saves nothing
  const local = ["localhost", "127.0.0.1", ""].includes(location.hostname);
  const banner = CONFIG.ENDPOINT ? "" : `<p class="banner">${local
    ? "TEST MODE — no Google Sheet connected (config.js ENDPOINT is empty). Data is downloaded as a file at the end."
    : "இந்த ஆய்வு இன்னும் தயாராகவில்லை. ஆய்வாளரைத் தொடர்பு கொள்ளவும். / This study is not set up yet (no data connection). Please contact the researcher."}</p>`;
  if (banner && !local) { show(`<div class="card"><h1>தமிழ் வாசிப்பு ஆய்வு</h1>${banner}</div>`); return; }
  show(`<div class="card">${banner}
    <h1>தமிழ் வாசிப்பு ஆய்வு</h1>
    <p class="en">Tamil Reading Study</p>
    <p>இந்த ஆய்வில் நீங்கள் <b>100 தமிழ் வாக்கியங்களைப்</b> படிப்பீர்கள். உங்கள் கண்கள் எங்கே பார்க்கின்றன என்பதை
       வெப்கேமரா மதிப்பிடும். சுமார் <b>30 நிமிடங்கள்</b> ஆகும். இது தேர்வு அல்ல.</p>
    <p class="en">You will read 100 Tamil sentences while the webcam estimates where you look (about 30 minutes).
       This is not a test. <b>No video or photos are saved</b> — only screen positions of your gaze.</p>

    <label for="pid">பங்கேற்பாளர் குறியீடு <span class="en">(Participant code from the researcher, e.g. R01)</span></label>
    <input type="text" id="pid" value="${esc(urlPid)}" maxlength="20" autocomplete="off">

    <label for="age">வயது <span class="en">(Age)</span></label>
    <input type="number" id="age" min="16" max="90">

    <label for="gender">பாலினம் <span class="en">(Gender)</span></label>
    <select id="gender"><option value="">—</option><option value="female">பெண் / Female</option>
      <option value="male">ஆண் / Male</option><option value="other">மற்றவை / Other</option>
      <option value="na">சொல்ல விரும்பவில்லை / Prefer not to say</option></select>

    <label for="native">தமிழ் உங்கள் தாய்மொழியா? <span class="en">(Is Tamil your first language?)</span></label>
    <select id="native"><option value="">—</option><option value="yes">ஆம் / Yes</option><option value="no">இல்லை / No</option></select>

    <label for="glasses">கண்ணாடி அணிகிறீர்களா? <span class="en">(Are you wearing glasses?)</span></label>
    <select id="glasses"><option value="">—</option><option value="no">இல்லை / No</option><option value="yes">ஆம் / Yes</option></select>

    <label class="check"><input type="checkbox" id="consent">
      <span>ஆய்வின் விவரங்களைப் படித்து, பங்கேற்க ஒப்புக்கொள்கிறேன்.
      <span class="en">I have read the study information and agree to take part.</span></span></label>
    <div class="error" id="err"></div>
    <button class="btn" id="start">தொடங்கு / Start</button></div>`);

  document.getElementById("start").onclick = () => {
    const v = id => document.getElementById(id).value.trim();
    const pid = v("pid").replace(/[^A-Za-z0-9_-]/g, "");
    if (!pid) { document.getElementById("err").textContent = "குறியீட்டை உள்ளிடவும் / Enter your participant code."; return; }
    if (!document.getElementById("consent").checked) {
      document.getElementById("err").textContent = "தொடர ஒப்புதல் தேவை / Please tick the consent box."; return;
    }
    state = {
      version: VERSION, pid, next: 0, practiceDone: false, fxno: 0, pending: [], results: [],
      info: { age: v("age"), gender: v("gender"), native_tamil: v("native"), glasses: v("glasses") },
      order: shuffled(STIMULI.map(s => s.sid), hashStr(pid)),
    };
    persist();
    record({ type: "participant", ...state.info, order: state.order.join(" "),
             gaze_source: MOUSE ? "mouse" : "webcam",
             screen_w: screen.width, screen_h: screen.height, window_w: innerWidth, window_h: innerHeight,
             dpr: devicePixelRatio, user_agent: navigator.userAgent });
    screenGuide();
  };
}

// Full instructions before the camera starts; every point must be ticked.
function screenGuide() {
  const ready = [
    ["மடிக்கணினியின் சார்ஜரை இணைத்துள்ளேன்; Zoom / Teams / Meet போன்ற கேமரா பயன்படுத்தும் செயலிகளை மூடியுள்ளேன்.",
     "My charger is plugged in and apps that use the camera (Zoom, Teams, Meet, Camera) are closed."],
    ["ஒளி என் முகத்தின் முன்னால் உள்ளது; எனக்குப் பின்னால் பிரகாசமான ஜன்னல் இல்லை.",
     "The light is in front of my face; there is no bright window behind me."],
    ["திரையிலிருந்து சுமார் ஒரு கை நீளத்தில், நேராக அமர்ந்துள்ளேன்; முழங்கைகளை மேசையில் வைத்து தாடையைக் கைகளில் தாங்கியுள்ளேன்.",
     "I sit straight, about an arm's length from the screen, elbows on the table and chin resting on my hands."],
    ["சுமார் 30 நிமிடங்கள் தடையின்றி இருக்க முடியும்; கைப்பேசியை அமைதியில் வைத்துள்ளேன்.",
     "I have about 30 minutes without interruptions; my phone is on silent."],
  ];
  show(`<div class="card"><h1>தொடங்கும் முன் / Before you start</h1>
    <h2>ஆய்வில் என்ன நடக்கும் / What will happen</h2>
    <ol>
      <li><b>கேமரா அமைப்பு (1 நிமிடம்)</b> — கேமராவை அனுமதித்து, உங்கள் முகம் கட்டத்துக்குள் பச்சையாகத் தெரியும்படி அமருங்கள்.
        <span class="en"><b>Camera (1 min)</b> — allow the camera and sit so your face is inside the box and the box is green.</span></li>
      <li><b>அளவீடு (2–3 நிமிடங்கள்)</b> — 13 சிவப்புப் புள்ளிகள் ஒவ்வொன்றாக வரும். புள்ளியைப் <b>பார்த்துக்கொண்டே</b> ${CONFIG.CAL_CLICKS} முறை கிளிக் செய்யவும்.
        பிறகு 9 மஞ்சள் புள்ளிகள் வரும்: அவற்றைப் பார்த்தால் மட்டும் போதும். துல்லியம் குறைவாக இருந்தால் மீண்டும் அளவீடு செய்யப்படும்.
        <span class="en"><b>Calibration (2–3 min)</b> — 13 red dots appear one by one. <b>Look at</b> each dot and click it ${CONFIG.CAL_CLICKS} times.
        Then 9 yellow dots appear: just look at them. If accuracy is low you calibrate again.</span></li>
      <li><b>பயிற்சி</b> — 3 பயிற்சி வாக்கியங்கள். <span class="en"><b>Practice</b> — 3 practice sentences.</span></li>
      <li><b>வாசிப்பு (சுமார் 20 நிமிடங்கள்)</b> — 100 வாக்கியங்கள், 4 பகுதிகள். ஒவ்வொரு பகுதிக்கும் பிறகு ஓய்வும் சுருக்கமான அளவீடும்.
        <span class="en"><b>Reading (about 20 min)</b> — 100 sentences in 4 parts, with a short break and a quick calibration after each part.</span></li>
    </ol>
    <h2>ஒவ்வொரு வாக்கியத்திலும் / For every sentence</h2>
    <table class="keys">
      <tr><td>①</td><td>இடப்பக்கம் மஞ்சள் புள்ளி வரும்: அதைப் பார்த்து <kbd>SPACE</kbd> அழுத்தவும்.
        <span class="en">A yellow dot appears on the left: look at it and press <kbd>SPACE</kbd>.</span></td></tr>
      <tr><td>②</td><td>முழு வாக்கியமும் தோன்றும் (1–4 வரிகள்). வழக்கம் போல இயல்பாகப் படியுங்கள். அவசரம் இல்லை.
        <span class="en">The whole sentence appears (1–4 lines). Read it normally, as you usually read. There is no hurry.</span></td></tr>
      <tr><td>③</td><td>படித்து முடித்ததும் <kbd>ENTER</kbd> அழுத்தவும். அப்போதுதான் அடுத்த வாக்கியம் வரும்.
        <span class="en">When you have finished, press <kbd>ENTER</kbd>. Only then the next sentence comes.</span></td></tr>
      <tr><td>④</td><td>சில நேரம் ஒரு சொல் காட்டப்படும்: அது வாக்கியத்தில் இருந்ததா? <kbd>Y</kbd> = ஆம், <kbd>N</kbd> = இல்லை.
        <span class="en">Sometimes a word is shown: was it in the sentence? <kbd>Y</kbd> = yes, <kbd>N</kbd> = no.</span></td></tr>
    </table>
    <h2>முக்கியம் / Important</h2>
    <ul>
      <li>முழு ஆய்வின்போதும் <b>தலையை அசைக்காதீர்கள்</b>; கண்களை மட்டும் நகர்த்துங்கள். ஓய்வின்போது மட்டும் அசையலாம்.
        <span class="en"><b>Keep your head still</b> the whole time and move only your eyes. You may move during the breaks.</span></li>
      <li>ஆய்வு முழுத்திரையில் இயங்கும். பக்கத்தைப் புதுப்பிக்கவோ மூடவோ வேண்டாம். தவறுதலாக மூடினால், அதே இணைப்பைத் திறந்தால் நிறுத்திய இடத்திலிருந்து தொடரலாம்.
        <span class="en">The study runs in full screen. Do not reload or close the page. If it closes by mistake, open the same link again to continue where you stopped.</span></li>
      <li>வீடியோ அல்லது புகைப்படம் எதுவும் சேமிக்கப்படாது; திரையில் நீங்கள் பார்க்கும் இடத்தின் எண்கள் மட்டுமே அனுப்பப்படும்.
        <span class="en">No video or photos are saved or sent — only numbers for where on the screen you look.</span></li>
    </ul>
    <h2>தயாரா? / Ready?</h2>
    ${ready.map(([ta, en], k) => `<label class="check"><input type="checkbox" class="ready" id="r${k}">
      <span>${ta} <span class="en">${en}</span></span></label>`).join("")}
    <div class="error" id="err"></div>
    <button class="btn" id="go">அடுத்து: கேமரா / Next: camera</button></div>`);
  document.getElementById("go").onclick = () => {
    if ([...document.querySelectorAll(".ready")].some(c => !c.checked)) {
      document.getElementById("err").textContent =
        "எல்லாவற்றையும் சரிபார்த்து டிக் செய்யவும் / Please check and tick every point.";
      return;
    }
    screenCamera();
  };
}

async function screenCamera() {
  show(`<div class="card"><h1>கேமரா / Camera</h1>
    <ol>
      <li>ஒளி உங்கள் முகத்தின் முன்னால் இருக்க வேண்டும் (பின்னால் ஜன்னல் வேண்டாம்).
        <span class="en">Light in front of your face, no bright window behind you.</span></li>
      <li>திரையிலிருந்து சுமார் ஒரு கை நீளத்தில் அமருங்கள். முழங்கைகளை மேசையில் வைத்து, தாடையைக் கைகளில் தாங்குங்கள்.
        <span class="en">Sit about an arm's length from the screen; rest your chin on your hands and keep your head still for the whole study.</span></li>
      <li>கேமரா அனுமதி கேட்டால் <b>Allow</b> அழுத்தவும்.
        <span class="en">When the browser asks for the camera, click <b>Allow</b>.</span></li>
    </ol>
    <div class="error" id="err"></div>
    <button class="btn" id="cam">கேமராவைத் தொடங்கு / Start camera</button></div>`);
  document.getElementById("cam").onclick = async () => {
    document.getElementById("cam").disabled = true;
    try {
      await startGaze();
    } catch (e) {
      document.getElementById("err").textContent =
        "கேமராவைத் திறக்க முடியவில்லை. அனுமதி கொடுத்து பக்கத்தை மீண்டும் ஏற்றவும். / The camera could not be opened. Allow camera access and reload the page.";
      document.getElementById("cam").disabled = false;
      return;
    }
    showCamera(true);
    show(`<div class="card"><h1>முகம் சரியாக உள்ளதா? / Is your face in the box?</h1>
      <p>இடது மேல் மூலையில் உங்கள் முகம் தெரியும். கட்டம் <b>பச்சை</b> நிறமாகும் வரை நகர்ந்து அமருங்கள்.</p>
      <p class="en">Your face appears in the top-left corner. Move until the box turns <b>green</b>, then keep that position.</p>
      <p>அடுத்து: திரையில் புள்ளிகள் வரும். ஒவ்வொரு புள்ளியையும் <b>பார்த்துக்கொண்டே</b> ${CONFIG.CAL_CLICKS} முறை கிளிக் செய்யவும்.</p>
      <p class="en">Next: dots appear one by one. <b>Look at each dot</b> and click it ${CONFIG.CAL_CLICKS} times while looking at it.</p>
      <button class="btn" id="go">அளவீடு தொடங்கு / Start calibration</button></div>`);
    document.getElementById("go").onclick = async () => {
      try { await document.documentElement.requestFullscreen(); } catch (e) { /* fullscreen not allowed */ }
      showCamera(false);
      await calibrateAndValidate("start", CONFIG.CAL_CLICKS);
      screenInstructions();
    };
  };
}

// ---------- calibration + validation ----------
const CAL_POINTS = [[.1, .1], [.5, .1], [.9, .1], [.3, .3], [.7, .3], [.1, .5], [.5, .5], [.9, .5],
                    [.3, .7], [.7, .7], [.1, .9], [.5, .9], [.9, .9]];
const VAL_POINTS = [[.2, .2], [.5, .2], [.8, .2], [.2, .5], [.5, .5], [.8, .5], [.2, .8], [.5, .8], [.8, .8]];

async function calibrate(clicks) {
  if (MOUSE) return;
  for (const [fx, fy] of CAL_POINTS) {
    const x = Math.round(fx * innerWidth), y = Math.round(fy * innerHeight);
    show(`<div class="overlay"><p class="corner">புள்ளியைப் பார்த்துக் கிளிக் செய்யவும் / Look at the dot and click it</p>
      <button class="caldot" id="dot" style="left:${x - 14}px;top:${y - 14}px"><span id="n">${clicks}</span></button></div>`);
    await new Promise(resolve => {
      let left = clicks;
      document.getElementById("dot").onclick = e => {
        webgazer.recordScreenPosition(x, y, "click");
        left -= 1;
        document.getElementById("n").textContent = left;
        if (left <= 0) resolve();
      };
    });
    await sleep(150);
  }
}

async function validate(label) {
  const errors = [], ex = [], ey = [];
  for (const [fx, fy] of VAL_POINTS) {
    const x = Math.round(fx * innerWidth), y = Math.round(fy * innerHeight);
    show(`<div class="overlay"><p class="corner">புள்ளியைப் பாருங்கள் / Look at the dot</p>
      <div class="valdot" style="left:${x - 12}px;top:${y - 12}px"></div></div>`);
    const t0 = now();
    await sleep(1500);
    const pts = recent.filter(s => s.valid && s.t >= t0 + 500);
    if (pts.length >= 3) {
      const mx = median(pts.map(p => p.x)), my = median(pts.map(p => p.y));
      errors.push(Math.hypot(mx - x, my - y)); ex.push(Math.abs(mx - x)); ey.push(Math.abs(my - y));
    }
  }
  const result = {
    type: "validation", label, points_with_data: errors.length,
    median_error_px: errors.length ? Math.round(median(errors)) : "",
    median_error_x_px: ex.length ? Math.round(median(ex)) : "", median_error_y_px: ey.length ? Math.round(median(ey)) : "",
    window_w: innerWidth, window_h: innerHeight,
  };
  record(result);
  return result;
}

async function calibrateAndValidate(label, clicks) {
  for (let attempt = 1; ; attempt++) {
    if (!MOUSE) await webgazer.clearData();
    await calibrate(clicks);
    const v = await validate(attempt === 1 ? label : `${label}_retry${attempt - 1}`);
    const good = v.points_with_data >= 6 && v.median_error_px !== "" && v.median_error_px <= CONFIG.MAX_ERROR_PX;
    if (good || MOUSE || attempt >= 3) return v;
    show(`<div class="card"><h2>துல்லியம் குறைவு / Accuracy is low</h2>
      <p>கண் கண்காணிப்பு போதுமான துல்லியமாக இல்லை (${v.median_error_px || "?"} px). ஒளியைச் சரிசெய்து, தலையை அசைக்காமல்
      மீண்டும் அளவீடு செய்வோம்.</p>
      <p class="en">Gaze accuracy was too low (${v.median_error_px || "?"} px, needs ≤ ${CONFIG.MAX_ERROR_PX}). Improve the light
      on your face, keep your head still, and calibrate again.</p>
      <button class="btn" id="again">மீண்டும் / Calibrate again</button></div>`);
    await new Promise(r => { document.getElementById("again").onclick = r; });
  }
}

// ---------- instructions ----------
function screenInstructions() {
  show(`<div class="card"><h1>வழிமுறைகள் / Instructions</h1>
    <ol>
      <li>ஒவ்வொரு வாக்கியத்திற்கும் முன் இடப்பக்கம் ஒரு புள்ளி வரும்: அதைப் பார்த்து <kbd>SPACE</kbd> அழுத்தவும்.
        <span class="en">Before each sentence a dot appears on the left: look at it and press <kbd>SPACE</kbd>.</span></li>
      <li>பிறகு முழு வாக்கியமும் தோன்றும். இயல்பாகப் படியுங்கள். படித்து முடித்ததும் <kbd>ENTER</kbd> அழுத்தவும்.
        <span class="en">Then the whole sentence appears. Read it normally. When you have finished, press <kbd>ENTER</kbd>.</span></li>
      <li>சில வாக்கியங்களுக்குப் பிறகு ஒரு சொல் காட்டப்படும்: அது வாக்கியத்தில் இருந்ததா? <kbd>Y</kbd> = ஆம், <kbd>N</kbd> = இல்லை.
        <span class="en">After some sentences a word is shown: was it in the sentence? <kbd>Y</kbd> = yes, <kbd>N</kbd> = no.</span></li>
      <li>தலையை அசைக்காமல் இருங்கள். 4 பகுதிகள்; ஒவ்வொன்றுக்கும் இடையில் ஓய்வு.
        <span class="en">Keep your head still. There are 4 parts with a short break between them.</span></li>
    </ol>
    <p>முதலில் 3 பயிற்சி வாக்கியங்கள். <span class="en">First, 3 practice sentences.</span></p>
    <button class="btn" id="go">பயிற்சி தொடங்கு / Start practice</button></div>`);
  document.getElementById("go").onclick = runSession;
}

// ---------- one sentence ----------
function layoutSentence(words) {
  // Word spans wrapped onto left-aligned lines, block centred vertically;
  // the font shrinks until the block fits the window.
  for (let size = CONFIG.FONT_PX; ; size -= 2) {
    show(`<div class="overlay"><p class="corner" id="hint"></p>
      <div class="text" id="text" style="font-size:${size}px;left:${CONFIG.MARGIN_PX}px;right:${CONFIG.MARGIN_PX}px">${
        words.map(w => `<span class="w" style="margin-right:${CONFIG.WORD_GAP_PX}px">${esc(w)}</span>`).join("")}</div></div>`);
    const text = document.getElementById("text");
    const first = text.querySelector(".w").getBoundingClientRect();
    text.style.lineHeight = `${Math.round(first.height * CONFIG.LINE_SPACING)}px`;
    text.style.top = `${Math.max(CONFIG.MARGIN_PX / 2, (innerHeight - text.offsetHeight) / 2)}px`;
    const boxes = [...text.querySelectorAll(".w")].map(el => {
      const r = el.getBoundingClientRect();
      return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)];
    });
    const fits = text.offsetHeight <= innerHeight - CONFIG.MARGIN_PX;
    if (fits || size <= CONFIG.MIN_FONT_PX) return { text, boxes, size };
  }
}

async function runTrial(sid, practice, trialNo) {
  const item = practice ? PRACTICE.find(p => p.sid === sid) : BY_SID[sid];
  const words = item.text.split(/\s+/);
  const { text, boxes, size } = layoutSentence(words);
  text.style.visibility = "hidden";

  // Drift check: look at the dot where the first word starts
  const dot = { x: boxes[0][0] + 8, y: Math.round((boxes[0][1] + boxes[0][3]) / 2) };
  const dotEl = document.createElement("div");
  dotEl.className = "fixdot";
  dotEl.style.left = `${dot.x - 10}px`; dotEl.style.top = `${dot.y - 10}px`;
  text.parentNode.appendChild(dotEl);
  const progress = practice ? "பயிற்சி / Practice"
    : `பகுதி ${Math.floor((trialNo - 1) / CONFIG.BLOCK_SIZE) + 1}/4 · ${trialNo}/${state.order.length}`;
  document.getElementById("hint").textContent =
    `${progress}  —  புள்ளியைப் பார்த்து SPACE அழுத்தவும் / Look at the dot and press SPACE`;
  const { t: tSpace } = await waitKey([" "], 300);
  const pts = recent.filter(s => s.valid && s.t >= tSpace - CONFIG.DRIFT_WINDOW_MS);
  offset = pts.length >= CONFIG.MIN_DRIFT_SAMPLES
    ? { x: median(pts.map(p => p.x)) - dot.x, y: median(pts.map(p => p.y)) - dot.y } : { x: 0, y: 0 };

  // Whole sentence until ENTER
  dotEl.remove();
  document.getElementById("hint").textContent = "";
  text.style.visibility = "visible";
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); // painted
  sink = [];
  const tOn = now();
  const { t: tOff } = await waitKey(["Enter"], CONFIG.MIN_TRIAL_MS);
  const samples = sink.filter(s => s.t <= tOff);
  sink = null;
  show(`<div class="overlay"></div>`);

  // Measures for this sentence, now
  const screenNo = practice ? 0 : SCREEN[sid];
  const table = Measures.wordTable(words, boxes, screenNo, practice ? 1 : FIRST_WNUM[sid], FREQ);
  const result = Measures.analyseSentence(samples, tOn, tOff, words, boxes, table, {
    dispersion: CONFIG.DISPERSION_PX, minDur: CONFIG.MIN_FIX_MS, maxGap: CONFIG.MAX_GAP_MS,
    maxFix: CONFIG.MAX_FIX_MS, lineBand: CONFIG.LINE_BAND_PX, gap: CONFIG.WORD_GAP_PX,
    minBlink: CONFIG.MIN_BLINK_MS, minValid: CONFIG.MIN_VALID, maxOffText: CONFIG.MAX_OFF_TEXT,
    fxnoStart: practice ? 0 : state.fxno,
  });

  // Comprehension question
  const answer = { probe_word: "", probe_answer: "", response: "", correct: "", answer_rt_ms: "" };
  if (item.probe) {
    show(`<div class="card"><h2>இந்தச் சொல் நீங்கள் இப்போது படித்த வாக்கியத்தில் இருந்ததா?</h2>
      <p class="en">Was this word in the sentence you just read?</p>
      <div class="probe-word">${esc(item.probe.word)}</div>
      <div class="row"><button class="btn" data-r="yes">ஆம் / Yes (Y)</button>
      <button class="btn secondary" data-r="no">இல்லை / No (N)</button></div></div>`);
    const t0 = now();
    const response = await new Promise(resolve => {
      document.querySelectorAll("[data-r]").forEach(b => { b.onclick = () => resolve(b.dataset.r); });
      waitKey(["y", "Y", "n", "N"]).then(k => resolve(k.key.toLowerCase() === "y" ? "yes" : "no"));
    });
    Object.assign(answer, { probe_word: item.probe.word, probe_answer: item.probe.answer, response,
                            correct: response === item.probe.answer ? 1 : 0, answer_rt_ms: Math.round(now() - t0) });
  }

  // Upload this sentence: one sentence row, a row per word, a row per event
  const q = result.quality;
  const sentence = {
    trial_no: trialNo, practice: practice ? 1 : 0, sid, band: item.band || "", screen: screenNo,
    n_words: words.length, n_lines: new Set(boxes.map(b => b[1])).size, font_px: size,
    reading_time_ms: Math.round(tOff - tOn), ...q,
    drift_x: Math.round(offset.x), drift_y: Math.round(offset.y), ...answer,
    window_w: innerWidth, window_h: innerHeight, boxes: JSON.stringify(boxes),
    raw_gaze: samples.map(s => s.valid ? `${Math.round(s.t - tOn)},${Math.round(s.x)},${Math.round(s.y)}`
                                       : `${Math.round(s.t - tOn)},,`).join(";"),
  };
  const payload = { type: "sentence", sentence };
  if (!practice) {
    payload.words = table.map((w, k) => ({
      trial_no: trialNo, sid, band: item.band || "", TEXT: w.screen, WNUM: w.wnum, WORD: w.word,
      LINE: w.line, LINE_POS: w.line_position, SERIAL: w.serial, START_POS: w.initial,
      OLEN: w.olen, WLEN: w.wlen, PUNCT: w.punct, TXFR: w.txfr,
      FFD: Math.round(result.measures[k].ffd), FPD: Math.round(result.measures[k].fpd),
      RPD: Math.round(result.measures[k].rpd), TD: Math.round(result.measures[k].td),
      N_FIX: result.measures[k].n_fix, FIXATED: result.measures[k].n_fix > 0 ? 1 : 0,
      SENTENCE_OK: q.sentence_ok,
    }));
    payload.fixations = result.events.map(e => ({ trial_no: trialNo, sid, ...e }));
    state.fxno = result.fxnoEnd;
  }
  record(payload);
}

// ---------- session ----------
async function restScreen(part) {
  show(`<div class="card"><h1>ஓய்வு / Break</h1>
    <p>பகுதி ${part} / 4 முடிந்தது. சிறிது ஓய்வெடுங்கள். பிறகு முன்பு போலவே அமர்ந்து, மீண்டும் அளவீடு செய்வோம்.</p>
    <p class="en">Part ${part} of 4 done. Rest a moment, sit as before, then we calibrate again briefly.</p>
    <button class="btn" id="go">தொடர்க / Continue</button></div>`);
  await new Promise(r => { document.getElementById("go").onclick = r; });
}

async function runSession() {
  if (!state.practiceDone) {
    for (let i = 0; i < PRACTICE.length; i++) await runTrial(PRACTICE[i].sid, true, -(PRACTICE.length - i));
    state.practiceDone = true;
    persist();
    show(`<div class="card"><h2>பயிற்சி முடிந்தது / Practice done</h2>
      <p>இப்போது உண்மையான வாக்கியங்கள் தொடங்கும். <span class="en">The real sentences start now.</span></p>
      <button class="btn" id="go">தொடங்கு / Start</button></div>`);
    await new Promise(r => { document.getElementById("go").onclick = r; });
  }
  while (state.next < state.order.length) {
    if (state.next > 0 && state.next % CONFIG.BLOCK_SIZE === 0 && !state.breakDone?.[state.next]) {
      await restScreen(state.next / CONFIG.BLOCK_SIZE);
      await calibrateAndValidate(`break${state.next / CONFIG.BLOCK_SIZE}`, CONFIG.RECAL_CLICKS);
      state.breakDone = { ...(state.breakDone || {}), [state.next]: true };
      persist();
    }
    await runTrial(state.order[state.next], false, state.next + 1);
    state.next += 1;
    persist();
  }
  await validate("end");
  record({ type: "participant_end", sentences: state.order.length });
  screenEnd();
}

async function screenEnd() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); } catch (e) { /* ignore */ }
  if (!MOUSE) { try { webgazer.end(); } catch (e) { /* ignore */ } }
  show(`<div class="card"><h1>முடிந்தது. நன்றி! / Finished. Thank you!</h1>
    <p id="status">தரவு அனுப்பப்படுகிறது… <span class="en">Sending your data…</span></p></div>`);
  if (!CONFIG.ENDPOINT) {
    document.getElementById("status").innerHTML =
      `<span class="en">Test mode (no ENDPOINT in config.js): your data file is being downloaded.</span>`;
    downloadData();
    return;
  }
  for (let i = 0; i < 30 && state.pending.length; i++) { await flush(); if (state.pending.length) await sleep(2000); }
  document.getElementById("status").innerHTML = state.pending.length
    ? `இணைய இணைப்பைச் சரிபார்த்து இந்தப் பக்கத்தைத் திறந்தே வைத்திருங்கள். <span class="en">Some data is still waiting to be sent —
       check your internet and keep this page open.</span>`
    : `உங்கள் தரவு சேமிக்கப்பட்டது. இந்தச் சாளரத்தை மூடலாம். <span class="en">Your data has been saved. You can close this window.</span>`;
  if (!state.pending.length) localStorage.removeItem(STORE_KEY);
}

screenWelcome();
