// Word-level eye-tracking measures for one sentence, computed in the browser
// right after the reader presses ENTER. Same steps and definitions as the
// desktop tool (TamilReadingStudy/analyze_reading.py, dundee_format.py) and
// the English Dundee data in phase 1:
//   1. running median of 3 samples against webcam jitter
//   2. I-DT fixations (spread <= dispersion px for >= minDur ms; a gap of
//      more than maxGap ms without gaze ends a fixation)
//   3. fixation -> nearest text line -> word on that line by x
//   4. noise filter: fixations longer than maxFix ms dropped; sentence
//      excluded when valid gaze < minValid or > maxOffText of fixation
//      time is off the text
//   5. FFD, FPD, RPD, TD per word; Dundee MA1 fields per fixation
"use strict";

(function (root) {
  const MEASURES = ["ffd", "fpd", "rpd", "td"];
  const PUNCT_CODES = { ".": 1, ",": 2, ";": 3, ":": 4, "?": 5, "!": 6 };
  const OPENING = "\"'‘“([";
  const CLOSING = ".,;:?!\"'’”)]";
  const MISSING = -99;

  // Visible letters: a base character plus its combining marks
  // (as count_graphemes in tamil_render.py).
  function graphemes(text) {
    let count = 0;
    for (const ch of text) {
      if (/\s/.test(ch)) continue;
      if (/[\p{Mn}\p{Mc}]/u.test(ch) && count) continue;
      count += 1;
    }
    return count;
  }

  function splitMarks(token) {
    let a = 0, b = token.length;
    while (a < b && OPENING.includes(token[a])) a++;
    let c = b;
    while (c > a && CLOSING.includes(token[c - 1])) c--;
    return { bare: token.slice(a, c), omarks: a, emarks: b - c, closing: token.slice(c) };
  }

  const median = v => {
    const s = v.slice().sort((p, q) => p - q), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  function runningMedian(values, size = 3) {
    if (values.length < size) return values.slice();
    const half = size >> 1;
    const padded = values.slice(0, half).concat(values, values.slice(-half));
    return values.map((_, i) => median(padded.slice(i, i + size)));
  }

  // I-DT on valid samples (t in ms). Returns [{start, end, x, y}].
  function detectFixations(t, x, y, dispersion, minDur, maxGap) {
    if (t.length < 2) return [];
    const diffs = t.slice(1).map((v, i) => v - t[i]);
    const interval = median(diffs);
    const out = [];
    let i = 0;
    while (i < t.length) {
      let j = i;
      while (j + 1 < t.length) {
        if (t[j + 1] - t[j] > maxGap) break;
        const xs = x.slice(i, j + 2), ys = y.slice(i, j + 2);
        if (Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) > dispersion) break;
        j += 1;
      }
      if (t[j] - t[i] + interval >= minDur) {
        out.push({ start: t[i], end: t[j] + interval,
                   x: median(x.slice(i, j + 1)), y: median(y.slice(i, j + 1)) });
        i = j + 1;
      } else {
        i += 1;
      }
    }
    return out;
  }

  function assignWord(fx, fy, boxes, gap, lineBand) {
    const centres = [...new Set(boxes.map(b => (b[1] + b[3]) / 2))].sort((p, q) => p - q);
    const lineY = centres.reduce((best, c) => Math.abs(fy - c) < Math.abs(fy - best) ? c : best);
    if (Math.abs(fy - lineY) > lineBand) return null;
    for (let k = 0; k < boxes.length; k++) {
      const [x1, y1, x2, y2] = boxes[k];
      if ((y1 + y2) / 2 === lineY && x1 - gap / 2 <= fx && fx <= x2 + gap / 2) return k;
    }
    return null;
  }

  // Gaps without valid gaze (blink, face lost): [{start, dur}]
  function gazeGaps(samples, tOff, minDur) {
    const gaps = [];
    let start = null;
    for (const s of samples) {
      if (!s.valid) { if (start === null) start = s.t; }
      else if (start !== null) { if (s.t - start >= minDur) gaps.push({ start, dur: s.t - start }); start = null; }
    }
    if (start !== null && tOff - start >= minDur) gaps.push({ start, dur: tOff - start });
    return gaps;
  }

  // Phase 1 definitions. sequence: [[word, durationMs]] in time order.
  function wordMeasures(sequence, nWords) {
    const result = [];
    for (let word = 0; word < nWords; word++) {
      const positions = [];
      sequence.forEach(([w], k) => { if (w === word) positions.push(k); });
      if (!positions.length) { result.push({ ffd: 0, fpd: 0, rpd: 0, td: 0, n_fix: 0 }); continue; }
      const durations = positions.map(k => sequence[k][1]);
      const first = positions[0];
      let fpd = durations[0];
      for (let k = first + 1; k < sequence.length; k++) {
        if (sequence[k][0] !== word) break;
        fpd += sequence[k][1];
      }
      let forward = -1;
      for (let k = first + 1; k < sequence.length; k++) if (sequence[k][0] > word) { forward = k; break; }
      const td = durations.reduce((a, b) => a + b, 0);
      const rpd = forward < 0 ? td
        : positions.reduce((a, k, n) => a + (k < forward ? durations[n] : 0), 0);
      result.push({ ffd: durations[0], fpd, rpd, td, n_fix: positions.length });
    }
    return result;
  }

  // Dundee per-word info (TX01WRDP fields) for one screen.
  // frequency: Map bare word -> count over all sentences.
  function wordTable(words, boxes, screen, firstWnum, frequency) {
    const tops = [...new Set(boxes.map(b => b[1]))].sort((p, q) => p - q);
    const position = {}, start = {};
    return words.map((token, k) => {
      const line = tops.indexOf(boxes[k][1]) + 1;
      position[line] = (position[line] || 0) + 1;
      const m = splitMarks(token);
      const olen = graphemes(token);
      // first word at 0, later words at the space before them (Are 0, tourists 4)
      const initial = start[line] || 0;
      start[line] = initial + olen + 1;
      return {
        word: token, bare: m.bare, screen, line, line_position: position[line], serial: k + 1,
        initial, olen, wlen: graphemes(m.bare),
        punct: m.closing ? (PUNCT_CODES[m.closing[0]] || 0) : 0,
        omarks: m.omarks, emarks: m.emarks, wnum: firstWnum + k,
        txfr: frequency.get(m.bare) || 0, box: boxes[k],
      };
    });
  }

  function landing(info, fx) {
    const [x1, , x2] = info.box;
    const oblp = fx < x1 ? 0 : Math.min(info.olen, Math.floor((fx - x1) / Math.max(x2 - x1, 1) * info.olen) + 1);
    const wdlp = Math.max(0, Math.min(oblp - info.omarks, info.wlen));
    return { oblp, wdlp, xpos: info.initial + oblp };
  }

  // Everything for one sentence. samples: [{t, x, y, valid}] while the
  // sentence was on screen (already drift-corrected); tOn/tOff in ms.
  // fxnoStart: number of events already written for this reader.
  function analyseSentence(samples, tOn, tOff, words, boxes, table, opt) {
    const valid = samples.filter(s => s.valid);
    const t = valid.map(s => s.t);
    let x = valid.map(s => s.x), y = valid.map(s => s.y);
    if (opt.smoothing !== false) { x = runningMedian(x); y = runningMedian(y); }

    const timeline = gazeGaps(samples, tOff, opt.minBlink).map(g => ({ kind: "blink", start: g.start, dur: g.dur }));
    const fixations = [];
    const sequence = [];
    let onText = 0, offText = 0, tooLong = 0;
    detectFixations(t, x, y, opt.dispersion, opt.minDur, opt.maxGap).forEach((f, number) => {
      const word = assignWord(f.x, f.y, boxes, opt.gap, opt.lineBand);
      const dur = f.end - f.start;
      const noise = dur > opt.maxFix ? "too_long" : word === null ? "off_text" : "";
      fixations.push({ number, start_ms: Math.round(f.start - tOn), duration_ms: Math.round(dur),
                       x: Math.round(f.x), y: Math.round(f.y), word_index: word === null ? "" : word, noise });
      if (noise === "too_long") { tooLong += 1; return; }
      if (word === null) offText += dur; else { onText += dur; sequence.push([word, dur]); }
      timeline.push({ kind: "fix", start: f.start, dur, x: f.x, word });
    });
    timeline.sort((p, q) => p.start - q.start);

    // Sentence quality
    const validShare = valid.length / Math.max(samples.length, 1);
    const offShare = offText / Math.max(onText + offText, 1);
    const problems = [];
    if (validShare < opt.minValid) problems.push("little valid gaze");
    if (offShare > opt.maxOffText) problems.push("gaze mostly off the text");
    if (!sequence.length) problems.push("no fixation on a word");

    // Dundee MA1 rows (events in time order)
    let fxno = opt.fxnoStart, previous = null, first = true;
    const events = timeline.map(e => {
      fxno += 1;
      const dur = Math.round(e.dur);
      if (e.kind === "blink" || e.word === null) {
        previous = null; first = false;
        return { FXNO: fxno, WORD: e.kind === "blink" ? "*Blink" : "*Offscreen", TEXT: table[0].screen,
                 LINE: MISSING, OLEN: MISSING, WLEN: MISSING, XPOS: MISSING, WNUM: MISSING, FDUR: dur,
                 OBLP: MISSING, WDLP: MISSING, LAUN: MISSING, TXFR: MISSING };
      }
      const info = table[e.word];
      const l = landing(info, e.x);
      // LAUN: previous XPOS minus this XPOS; 0 first on screen; -99 after a blink or line change
      const laun = first ? 0 : previous === null || previous.line !== info.line ? MISSING : previous.xpos - l.xpos;
      previous = { line: info.line, xpos: l.xpos }; first = false;
      return { FXNO: fxno, WORD: info.word, TEXT: info.screen, LINE: info.line, OLEN: info.olen,
               WLEN: info.wlen, XPOS: l.xpos, WNUM: info.wnum, FDUR: dur, OBLP: l.oblp, WDLP: l.wdlp,
               LAUN: laun, TXFR: info.txfr };
    });

    return {
      measures: wordMeasures(sequence, words.length),
      fixations, events, fxnoEnd: fxno,
      quality: {
        samples: samples.length, valid_gaze_pct: +(100 * validShare).toFixed(1),
        fixations: sequence.length, too_long_dropped: tooLong, off_text_pct: +(100 * offShare).toFixed(1),
        sentence_ok: problems.length ? 0 : 1, excluded_because: problems.join("; "),
      },
    };
  }

  const api = { MEASURES, graphemes, splitMarks, runningMedian, detectFixations, assignWord,
                gazeGaps, wordMeasures, wordTable, landing, analyseSentence };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Measures = api;
})(typeof window !== "undefined" ? window : globalThis);
