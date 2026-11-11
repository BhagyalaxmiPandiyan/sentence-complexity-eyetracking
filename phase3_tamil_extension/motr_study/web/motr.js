// MoTR (Mouse Tracking for Reading, Wilcox et al. 2024) for Tamil.
// The sentence is blurred; only the area around the mouse is clear. Readers move the
// mouse along the words as they read, so the mouse path stands in for the eye path.
// Each trial records every mouse sample and every word "visit" (= one fixation),
// which compute_measures.py turns into FFD / FPD / RPD / TD exactly like Dundee.
"use strict";

const VERSION = "motr-ta-1.0";
const STORE_KEY = "motr_ta_state";
const $app = document.getElementById("app");

// ---------- small helpers ----------
const load = k => { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage full or blocked */ } };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = html => { $app.innerHTML = html; window.scrollTo(0, 0); };

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
// Same participant code -> same order (so a reload resumes correctly); different codes -> different orders.
function shuffled(arr, seed) {
  const rnd = mulberry32(seed), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

// ---------- state + upload queue (survives a page reload) ----------
let state = null; // {pid, info, order:[sid], next, practiceDone, results:[], pending:[]}
const persist = () => save(STORE_KEY, state);

async function flush() {
  if (!CONFIG.ENDPOINT) return state.pending.length === 0;
  while (state.pending.length) {
    try {
      await fetch(CONFIG.ENDPOINT, {
        method: "POST", mode: "no-cors",
        headers: { "Content-Type": "text/plain;charset=utf-8" },
        body: JSON.stringify(state.pending[0]),
      });
      state.pending.shift();
      persist();
    } catch (e) {
      return false;
    }
  }
  return true;
}
function record(payload) {
  payload = { version: VERSION, pid: state.pid, ts: new Date().toISOString(), ...payload };
  state.results.push(payload);
  if (CONFIG.ENDPOINT) state.pending.push(payload);
  persist();
  flush();
}
function downloadData() {
  const blob = new Blob([JSON.stringify({ pid: state.pid, info: state.info, results: state.results })],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `motr_tamil_${state.pid}.json`;
  a.click();
}

// ---------- screens ----------
function deviceOK() {
  return window.matchMedia("(pointer: fine)").matches && window.innerWidth >= 900;
}

function screenWelcome() {
  if (!deviceOK()) {
    show(`<div class="card"><h1>தமிழ் வாசிப்பு ஆய்வு</h1>
      <p>இந்த ஆய்வை <b>கணினியில் (லேப்டாப் / டெஸ்க்டாப்) மவுஸ் அல்லது டச்பேட்</b> கொண்டு மட்டுமே செய்ய முடியும்.
      கைப்பேசி அல்லது டேப்லெட்டில் செய்ய முடியாது. உலாவி சாளரம் சிறியதாக இருந்தால், அதைப் பெரிதாக்கி பக்கத்தை மீண்டும் ஏற்றவும்.</p>
      <p class="en">This study needs a laptop or desktop with a mouse or touchpad, and a browser window at least 900px wide.
      Phones and tablets are not supported.</p></div>`);
    return;
  }
  const saved = load(STORE_KEY);
  if (saved && saved.pid && saved.next < saved.order.length) {
    show(`<div class="card"><h1>மீண்டும் வருக!</h1>
      <p>குறியீடு <b>${esc(saved.pid)}</b> — ${saved.next} / ${saved.order.length} வாக்கியங்கள் முடிந்துள்ளன. நிறுத்திய இடத்திலிருந்து தொடரலாமா?</p>
      <p class="en">You already started (code ${esc(saved.pid)}, ${saved.next}/${saved.order.length} done). Continue where you stopped?</p>
      <button class="btn" id="resume">தொடர்க / Continue</button>
      <button class="btn secondary" id="fresh">புதிதாகத் தொடங்கு / Start new</button></div>`);
    document.getElementById("resume").onclick = () => { state = saved; flush(); screenInstructions(true); };
    document.getElementById("fresh").onclick = () => {
      if (confirm("முந்தைய முன்னேற்றம் நீக்கப்படும். உறுதியா? / Previous progress will be cleared. Sure?")) {
        localStorage.removeItem(STORE_KEY); screenWelcome();
      }
    };
    return;
  }

  const urlPid = new URLSearchParams(location.search).get("pid") || "";
  show(`<div class="card">
    <h1>தமிழ் வாசிப்பு ஆய்வு</h1>
    <p class="en">Tamil Reading Study</p>
    <p>இந்த ஆய்வில் நீங்கள் <b>100 தமிழ் செய்தித்தாள் வாக்கியங்களைப்</b> படிப்பீர்கள். சுமார் <b>25–30 நிமிடங்கள்</b> ஆகும்.
       இது தேர்வு அல்ல; சரி / தவறு என்று எதுவும் இல்லை. உங்கள் இயல்பான வாசிப்பு முறையை அறிவதே நோக்கம்.</p>
    <p class="en">You will read 100 Tamil newspaper sentences (about 25–30 minutes). This is not a test — we only want to see how people naturally read.</p>

    <label for="pid">பங்கேற்பாளர் குறியீடு <span class="en">(Participant code, given by the researcher, e.g. R01)</span></label>
    <input type="text" id="pid" value="${esc(urlPid)}" maxlength="20" autocomplete="off">

    <label for="age">வயது <span class="en">(Age)</span></label>
    <input type="number" id="age" min="16" max="90">

    <label for="gender">பாலினம் <span class="en">(Gender)</span></label>
    <select id="gender"><option value="">—</option><option value="female">பெண் / Female</option>
      <option value="male">ஆண் / Male</option><option value="other">மற்றவை / Other</option>
      <option value="na">சொல்ல விரும்பவில்லை / Prefer not to say</option></select>

    <label for="native">தமிழ் உங்கள் தாய்மொழியா? <span class="en">(Is Tamil your first language?)</span></label>
    <select id="native"><option value="">—</option><option value="yes">ஆம் / Yes</option><option value="no">இல்லை / No</option></select>

    <label for="edu">கல்வித் தகுதி <span class="en">(Highest education)</span></label>
    <select id="edu"><option value="">—</option><option value="school">பள்ளி / School</option>
      <option value="ug">இளநிலைப் பட்டம் / UG degree</option><option value="pg">முதுநிலைப் பட்டம் / PG degree</option>
      <option value="phd">முனைவர் / PhD</option></select>

    <label for="news">தமிழ் செய்தித்தாளை எவ்வளவு அடிக்கடி படிப்பீர்கள்? <span class="en">(How often do you read Tamil newspapers?)</span></label>
    <select id="news"><option value="">—</option><option value="daily">தினமும் / Daily</option>
      <option value="weekly">வாரத்தில் சில முறை / A few times a week</option><option value="rarely">அரிதாக / Rarely</option></select>

    <label class="check"><input type="checkbox" id="consent">
      <span>நான் இந்த ஆய்வில் விருப்பத்துடன் பங்கேற்கிறேன். என் பெயர் சேமிக்கப்படாது; என் வாசிப்புத் தரவு ஆய்வுக்கு மட்டுமே பயன்படும். எப்போது வேண்டுமானாலும் நான் விலகலாம்.
      <br><span class="en">I agree to take part. My name is not stored; my reading data is used only for research. I can stop at any time.</span></span></label>

    <div id="err" class="error"></div>
    <button class="btn" id="go">தொடங்கு / Start</button>
  </div>`);

  document.getElementById("go").onclick = () => {
    const v = id => document.getElementById(id).value.trim();
    const pid = v("pid").toUpperCase().replace(/[^A-Z0-9_-]/g, "");
    const err = document.getElementById("err");
    if (!pid) return (err.textContent = "குறியீட்டை உள்ளிடவும் / Please enter your participant code.");
    if (!v("age") || !v("native") || !v("news")) return (err.textContent = "எல்லா விவரங்களையும் நிரப்பவும் / Please fill in all details.");
    if (!document.getElementById("consent").checked) return (err.textContent = "ஒப்புதல் பெட்டியைத் தேர்வு செய்யவும் / Please tick the consent box.");
    const info = { age: v("age"), gender: v("gender"), native_tamil: v("native"), education: v("edu"), news_reading: v("news") };
    state = {
      pid, info, next: 0, practiceDone: false, results: [], pending: [],
      order: shuffled(STIMULI.map(s => s.sid), hashStr(pid)),
    };
    record({
      type: "participant", ...info, order: state.order.join(" "),
      screen_w: screen.width, screen_h: screen.height, window_w: innerWidth, window_h: innerHeight,
      dpr: devicePixelRatio, user_agent: navigator.userAgent,
    });
    screenInstructions(false);
  };
}

function screenInstructions(resuming) {
  show(`<div class="card">
    <h2>எப்படிப் படிப்பது? <span class="en">How to read</span></h2>
    <ol>
      <li>திரையில் ஒரு வாக்கியம் <b>மங்கலாக</b> தெரியும். இடப்பக்கம் உள்ள <b>பச்சைப் புள்ளியைக் கிளிக்</b> செய்து தொடங்குங்கள்.</li>
      <li>மவுஸை சொற்களின் மேல் நகர்த்துங்கள். <b>மவுஸ் இருக்கும் இடம் மட்டும் தெளிவாகத்</b> தெரியும்.</li>
      <li>உங்கள் <b>கண்கள் படிக்கும் இடத்திலேயே மவுஸை</b> வைத்து, வழக்கம் போல <b>இயல்பான வேகத்தில்</b> படியுங்கள். தேவைப்பட்டால் பின்னோக்கிச் சென்று மீண்டும் படிக்கலாம்.</li>
      <li>படித்து முடித்ததும் <kbd>Space</kbd> விசையை அழுத்துங்கள்.</li>
      <li>சில வாக்கியங்களுக்குப் பிறகு, ஒரு சொல் அந்த வாக்கியத்தில் <b>இருந்ததா இல்லையா</b> என்று கேட்கப்படும்.</li>
    </ol>
    <p class="en">The sentence is blurred. Click the green dot, then move the mouse along the words — only the part under the mouse
      becomes clear. Keep the mouse where your eyes are reading, at your normal speed; you may go back and re-read. Press
      <kbd>Space</kbd> when you finish. Sometimes we will ask whether a word was in the sentence.</p>
    <p>அமைதியான இடத்தில், இடையூறு இல்லாமல் செய்யவும். <span class="en">Please do this in a quiet place without interruptions.</span></p>
    <button class="btn" id="go">${resuming || state.practiceDone ? "தொடர்க / Continue" : "பயிற்சியைத் தொடங்கு / Start practice"}</button>
  </div>`);
  document.getElementById("go").onclick = () => {
    try { document.documentElement.requestFullscreen(); } catch (e) { /* optional */ }
    state.practiceDone ? mainLoop() : practiceLoop();
  };
}

async function practiceLoop() {
  for (let i = 0; i < PRACTICE.length; i++) {
    const item = { ...PRACTICE[i] };
    if (i === 1) { // one demo question so the first real question is not a surprise
      const longest = item.text.split(/\s+/).map(w => w.replace(/[.,]/g, "")).sort((a, b) => b.length - a.length)[0];
      item.probe = { word: longest, answer: "yes" };
    }
    const res = await runTrial(item, `பயிற்சி / Practice ${i + 1} / ${PRACTICE.length}`);
    record({ type: "trial", practice: 1, trial_no: i + 1, ...res });
  }
  state.practiceDone = true;
  persist();
  show(`<div class="card"><h2>பயிற்சி முடிந்தது! <span class="en">Practice done</span></h2>
    <p>இப்போது உண்மையான ஆய்வு தொடங்குகிறது: 100 வாக்கியங்கள். பாதியில் ஓய்வு எடுக்கலாம்.</p>
    <p class="en">Now the real study begins: 100 sentences, with a break halfway.</p>
    <button class="btn" id="go">தொடங்கு / Begin</button></div>`);
  document.getElementById("go").onclick = mainLoop;
}

async function mainLoop() {
  const byId = Object.fromEntries(STIMULI.map(s => [s.sid, s]));
  const n = state.order.length;
  while (state.next < n) {
    if (state.next === CONFIG.BREAK_AFTER) await screenBreak();
    const item = byId[state.order[state.next]];
    const res = await runTrial(item, `${state.next + 1} / ${n}`);
    record({ type: "trial", practice: 0, trial_no: state.next + 1, ...res });
    state.next += 1;
    persist();
  }
  screenEnd();
}

function screenBreak() {
  return new Promise(resolve => {
    show(`<div class="card"><h2>பாதி முடிந்தது! <span class="en">Halfway there</span></h2>
      <p>சிறிது ஓய்வு எடுத்துக்கொள்ளுங்கள். தயாரானதும் தொடருங்கள்.</p>
      <p class="en">Take a short rest. Continue when you are ready.</p>
      <button class="btn" id="go">தொடர்க / Continue</button></div>`);
    document.getElementById("go").onclick = resolve;
  });
}

async function screenEnd() {
  show(`<div class="card"><h1>நன்றி! <span class="en">Thank you!</span></h1><p id="status">தரவு அனுப்பப்படுகிறது… / Sending data…</p></div>`);
  try { if (document.fullscreenElement) document.exitFullscreen(); } catch (e) { /* ignore */ }
  const ok = CONFIG.ENDPOINT ? await flush() : false;
  const probes = state.results.filter(r => r.type === "trial" && !r.practice && r.probe_word);
  const acc = probes.length ? probes.filter(r => r.correct).length / probes.length : null;
  const $s = document.getElementById("status");
  if (ok) {
    $s.innerHTML = `ஆய்வு முடிந்தது. உங்கள் தரவு வெற்றிகரமாக அனுப்பப்பட்டது. இந்தப் பக்கத்தை மூடலாம்.
      <br><span class="en">Done — your data was sent. You can close this page.</span>`;
    localStorage.removeItem(STORE_KEY);
  } else {
    $s.innerHTML = `${CONFIG.ENDPOINT ? "தரவை தானாக அனுப்ப முடியவில்லை." : "சோதனை முறை (test mode)."}
      கீழே உள்ள பொத்தானை அழுத்தி கோப்பைப் பதிவிறக்கி, ஆய்வாளருக்கு அனுப்புங்கள்.
      <br><span class="en">Please download the file below and send it to the researcher.</span>
      <br><button class="btn" id="dl">தரவைப் பதிவிறக்கு / Download data</button>`;
    document.getElementById("dl").onclick = downloadData;
  }
  if (acc !== null) console.log(`Probe accuracy: ${(acc * 100).toFixed(0)}%`);
}

// ---------- one reading trial ----------
// Interest areas (AOIs), like eye-tracking: each word's box is widened to the middle of the
// gap to its neighbours and stretched vertically to the middle of the gap between lines,
// so every point on the text belongs to exactly one word.
function computeAOIs(stage) {
  const sr = stage.getBoundingClientRect();
  const boxes = [...stage.querySelectorAll(".clear .w")].map(s => {
    const r = s.getBoundingClientRect();
    return { l: r.left - sr.left, r: r.right - sr.left, t: r.top - sr.top, b: r.bottom - sr.top };
  });
  const lines = [];
  boxes.forEach((b, i) => {
    let L = lines.find(L => Math.abs(L.t - b.t) < 6);
    if (!L) { L = { t: b.t, b: b.b, idx: [] }; lines.push(L); }
    L.idx.push(i);
    L.b = Math.max(L.b, b.b);
  });
  lines.sort((a, b) => a.t - b.t);
  const aois = new Array(boxes.length);
  lines.forEach((L, li) => {
    const h = L.b - L.t;
    const y0 = li === 0 ? L.t - 0.6 * h : (lines[li - 1].b + L.t) / 2;
    const y1 = li === lines.length - 1 ? L.b + 0.9 * h : (L.b + lines[li + 1].t) / 2;
    L.idx.forEach((i, k) => {
      const prev = k > 0 ? boxes[L.idx[k - 1]] : null;
      const next = k < L.idx.length - 1 ? boxes[L.idx[k + 1]] : null;
      aois[i] = {
        x0: Math.round(prev ? (prev.r + boxes[i].l) / 2 : boxes[i].l - 16),
        x1: Math.round(next ? (boxes[i].r + next.l) / 2 : boxes[i].r + 16),
        y0: Math.round(y0), y1: Math.round(y1), line: li,
      };
    });
  });
  return aois;
}

function runTrial(item, label) {
  return new Promise(async resolve => {
    const words = item.text.trim().split(/\s+/);
    const spans = words.map((w, i) => `<span class="w" data-i="${i}">${esc(w)}</span>`).join(" ");
    show(`<div class="trial-top"><span>${esc(label)}</span><span class="en">Space = முடிந்தது / done</span></div>
      <div id="stage" class="stage">
        <div class="text blurred" style="--blur:${CONFIG.BLUR_PX}px">${spans}</div>
        <div class="text clear" id="clear" style="--rx:${CONFIG.FOCUS_RX}px;--ry:${CONFIG.FOCUS_RY}px">${spans}</div>
        <button class="startdot" id="dot" aria-label="start"></button>
      </div>
      <p class="hint" id="hint">பச்சைப் புள்ளியைக் கிளிக் செய்து படிக்கத் தொடங்குங்கள் <span class="en">· click the green dot to start</span></p>`);

    await document.fonts.ready;
    const stage = document.getElementById("stage"), clear = document.getElementById("clear");
    const dot = document.getElementById("dot"), hint = document.getElementById("hint");
    let aois = computeAOIs(stage);
    dot.style.left = `${aois[0].x0 - 34}px`;
    dot.style.top = `${(aois[0].y0 + aois[0].y1) / 2 - 11}px`;
    const clearOff = { x: clear.offsetLeft, y: clear.offsetTop };

    let t0 = null, cur = { w: -1, start: 0 }, resized = 0, raf = 0, last = null;
    const visits = [], raw = [];
    const hit = (x, y) => aois.findIndex(a => x >= a.x0 && x < a.x1 && y >= a.y0 && y < a.y1);

    const onMove = e => {
      if (t0 === null) return;
      const sr = stage.getBoundingClientRect();
      const t = Math.round(e.timeStamp - t0), x = Math.round(e.clientX - sr.left), y = Math.round(e.clientY - sr.top);
      raw.push(`${t},${x},${y}`);
      const w = hit(x, y);
      if (w !== cur.w) {
        if (cur.w >= 0) visits.push([cur.w, cur.start, t - cur.start]);
        cur = { w, start: t };
      }
      last = { x, y };
      if (!raf) raf = requestAnimationFrame(() => {
        raf = 0;
        clear.style.setProperty("--mx", `${last.x - clearOff.x}px`);
        clear.style.setProperty("--my", `${last.y - clearOff.y}px`);
      });
    };
    const onResize = () => { resized = 1; aois = computeAOIs(stage); };

    const finish = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
      const t = Math.round(performance.now() - t0);
      if (cur.w >= 0) visits.push([cur.w, cur.start, t - cur.start]);
      const base = {
        sid: item.sid, text: item.text, n_words: words.length, rt_ms: t,
        aois: JSON.stringify(aois), visits: JSON.stringify(visits), raw: raw.join(";"),
        stage_w: stage.clientWidth, stage_h: stage.clientHeight, resized,
        probe_word: "", probe_answer: "", response: "", correct: "", probe_rt_ms: "",
      };
      if (!item.probe) return resolve(base);
      askProbe(item.probe).then(p => resolve({ ...base, ...p }));
    };
    const onKey = e => {
      if (e.code !== "Space") return;
      e.preventDefault();
      if (t0 === null) return;
      if (performance.now() - t0 < CONFIG.MIN_TRIAL_MS) return;
      if (visits.length === 0 && cur.w < 0) {
        hint.innerHTML = `மவுஸை சொற்களின் மேல் நகர்த்திப் படியுங்கள் <span class="en">· please move the mouse along the words while reading</span>`;
        return;
      }
      finish();
    };

    dot.onclick = e => {
      t0 = e.timeStamp;
      dot.remove();
      stage.classList.add("reading");
      hint.innerHTML = `படித்து முடித்ததும் <kbd>Space</kbd> அழுத்துங்கள் <span class="en">· press Space when you finish</span>`;
      onMove(e);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onResize);
  });
}

function askProbe(probe) {
  return new Promise(resolve => {
    show(`<div class="card">
      <h2>இந்தச் சொல் நீங்கள் இப்போது படித்த வாக்கியத்தில் இருந்ததா?</h2>
      <p class="en">Was this word in the sentence you just read?</p>
      <div class="probe-word">${esc(probe.word)}</div>
      <div class="row"><button class="btn" data-r="yes">ஆம் / Yes</button><button class="btn" data-r="no">இல்லை / No</button></div>
    </div>`);
    const t = performance.now();
    $app.querySelectorAll("[data-r]").forEach(b => b.onclick = () => resolve({
      probe_word: probe.word, probe_answer: probe.answer, response: b.dataset.r,
      correct: b.dataset.r === probe.answer ? 1 : 0, probe_rt_ms: Math.round(performance.now() - t),
    }));
  });
}

screenWelcome();
