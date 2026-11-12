"""
Look and story of the toolkit: CSS, small HTML builders (bars, cards, word chips) and the
static explanation pages. All numbers on the static pages come from the thesis results
(phase1_replication/results/summary.md, phase3_tamil_extension/results/*.md).
"""
import html

BLUE, ORANGE, INK, MUTED = "#1F4E8C", "#B45309", "#1F2933", "#52606D"

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
.tk * {box-sizing: border-box;}
.tk {font-family: 'Segoe UI', Lato, Arial, sans-serif; color: #1F2933; line-height: 1.45;}
.tk h2 {font-size: 26px; margin: 4px 0 8px; color: #1F2933;}
.tk h3 {font-size: 19px; margin: 14px 0 8px; color: #1F2933;}
.tk p {margin: 4px 0 8px; font-size: 16px;}
.tk .lead {font-size: 18px; color: #3E4C59;}
.tk .muted {color: #52606D; font-size: 14px;}
.tk .grid {display: grid; gap: 14px;}
.tk .g2 {grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));}
.tk .g3 {grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));}
.tk .g4 {grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));}
.tk .card {background: #F7F9FB; border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px 16px;}
.tk .card.blue {background: #EEF3FA; border-color: #C9D8EE;}
.tk .card.orange {background: #FDF3E7; border-color: #F0D9BC;}
.tk .card.green {background: #EAF6EE; border-color: #BFE3CB;}
.tk .tag {display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: .5px;
          text-transform: uppercase; padding: 2px 8px; border-radius: 6px; color: #fff; background: #1F4E8C;}
.tk .tag.orange {background: #B45309;} .tk .tag.grey {background: #7B8794;} .tk .tag.green {background: #1E7B45;}
.tk .big {font-size: 30px; font-weight: 700; color: #1F4E8C; line-height: 1.1;}
.tk .big.orange {color: #B45309;}
.tk .flow {display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px;}
.tk .flow .step {flex: 1 1 170px; background: #EEF3FA; border: 1px solid #C9D8EE; border-radius: 10px; padding: 10px 12px;}
.tk .flow .step.orange {background: #FDF3E7; border-color: #F0D9BC;}
.tk .flow .arrow {align-self: center; color: #7B8794; font-size: 22px;}
.tk .num {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; background: #B45309; color: #fff;
          font-weight: 700; align-items: center; justify-content: center; margin-right: 8px; font-size: 14px;}
.tk .num.blue {background: #1F4E8C;}
.tk table.t {border-collapse: collapse; width: 100%; font-size: 15px; background: #fff;}
.tk table.t th {background: #EEF3FA; text-align: left; padding: 7px 10px; border: 1px solid #D9E2EC; color: #1F2933;}
.tk table.t td {padding: 6px 10px; border: 1px solid #D9E2EC; color: #1F2933;}
.tk table.t td.c, .tk table.t th.c {text-align: center;}
.tk .barrow {display: grid; grid-template-columns: 150px 1fr 70px; gap: 10px; align-items: center; margin: 3px 0;}
.tk .barlabel {font-size: 15px; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.tk .bartrack {background: #EEF1F4; border-radius: 5px; height: 14px; position: relative;}
.tk .bar {height: 14px; border-radius: 5px;}
.tk .barval {font-size: 14px; color: #3E4C59; font-variant-numeric: tabular-nums;}
.tk .legend span {display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; font-size: 14px;}
.tk .legend i {display: inline-block; width: 14px; height: 10px; border-radius: 3px;}
.tk .verdict {font-size: 22px; font-weight: 700; padding: 12px 16px; border-radius: 10px;
              background: #EEF3FA; border-left: 6px solid #1F4E8C; margin: 6px 0 10px;}
.tk .verdict.orange {background: #FDF3E7; border-left-color: #B45309;}
.tk .levels {display: flex; gap: 6px; margin-top: 6px;}
.tk .lv {flex: 1; text-align: center; padding: 8px 4px; border-radius: 8px; background: #EEF1F4; color: #7B8794;
         font-weight: 600; font-size: 14px;}
.tk .lv.on {background: #1F4E8C; color: #fff;}
.tk .lv.on.orange {background: #B45309;}
.tk .words {display: flex; flex-wrap: wrap; gap: 10px;}
.tk .w {background: #fff; border: 1px solid #D9E2EC; border-radius: 10px; padding: 8px 10px; min-width: 150px;}
.tk .w .wt {font-size: 20px; font-weight: 700; font-family: 'Noto Sans Tamil', 'Latha', 'Nirmala UI', sans-serif;}
.tk .chips {display: flex; gap: 3px; margin: 6px 0; flex-wrap: wrap;}
.tk .chip {font-family: 'Noto Sans Tamil', 'Latha', 'Nirmala UI', sans-serif; font-size: 15px; padding: 2px 7px;
           border-radius: 6px; color: #fff;}
.tk .chip.stem {background: #1F4E8C;} .tk .chip.suf {background: #B45309;}
.tk .kv {font-size: 13px; color: #3E4C59; margin: 1px 0;}
.tk .kv b {color: #1F2933;}
.tk .ta {font-family: 'Noto Sans Tamil', 'Latha', 'Nirmala UI', sans-serif;}
.tk .note {background: #FFF8E6; border: 1px solid #F3DFA2; border-radius: 10px; padding: 10px 14px; font-size: 15px;}
"""

# Opens the page in light mode, so the coloured cards look the same on any projector.
FORCE_LIGHT = """() => { const u = new URL(window.location); if (u.searchParams.get('__theme') !== 'light') {
  u.searchParams.set('__theme', 'light'); window.location.replace(u); } }"""


def esc(x):
    return html.escape(str(x))


def wrap(body):
    return f'<div class="tk">{body}</div>'


def bars(labels, series, unit="", fmt="{:.0f}"):
    """Horizontal bar chart. series = [(name, values, colour), ...]; one row per label and series."""
    top = max([max(v) for _, v, _ in series if len(v)] + [1e-9])
    legend = "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, _, c in series)
    rows = []
    for i, lab in enumerate(labels):
        for k, (n, vals, c) in enumerate(series):
            v = float(vals[i])
            rows.append(
                f'<div class="barrow"><div class="barlabel">{esc(lab) if k == 0 else ""}</div>'
                f'<div class="bartrack"><div class="bar" style="width:{max(v, 0) / top * 100:.1f}%;background:{c}"></div></div>'
                f'<div class="barval">{fmt.format(v)}{unit}</div></div>')
    leg = f'<div class="legend">{legend}</div>' if len(series) > 1 else ""
    return leg + "".join(rows)


def levels(active, colour=""):
    return '<div class="levels">' + "".join(
        f'<div class="lv {"on " + colour if i == active else ""}">Level {i}</div>' for i in range(1, 5)) + "</div>"


def word_cards(df):
    """One card per Tamil word: Morfessor parts (stem blue, suffixes orange) + the numbers behind it."""
    out = []
    for _, r in df.iterrows():
        parts = str(r["Morphemes (Morfessor)"]).split(" + ")
        chips = "".join(f'<span class="chip {"stem" if i == 0 else "suf"}">{esc(p)}</span>' for i, p in enumerate(parts))
        nf = ' · <b>non-finite verb</b>' if r["Non-finite"] == "yes" else ""
        out.append(
            f'<div class="w"><div class="wt">{esc(r["Word"])}</div><div class="chips">{chips}</div>'
            f'<div class="kv"><b>{r["Letters"]}</b> letters · freq <b>{r["Word freq."]}</b></div>'
            f'<div class="kv">GPT-2 <b>{r["GPT-2 surpr."]}</b> bits (stem {r["Stem surpr."]} + suffix {r["Suffix surpr."]})</div>'
            f'<div class="kv">root <span class="ta"><b>{esc(r["Root (Stanza)"])}</b></span> · freq {r["Root freq."]}</div>'
            f'<div class="kv">marks: <b>{esc(r["Grammar marks"])}</b>{nf}</div>'
            f'<div class="kv">→ <span class="ta">{esc(r["Depends on"])}</span> · depth {r["Depth"]}</div></div>')
    legend = ('<div class="legend" style="margin:6px 0 10px"><span><i style="background:#1F4E8C"></i>stem (first part)</span>'
              '<span><i style="background:#B45309"></i>suffix parts</span></div>')
    return legend + '<div class="words">' + "".join(out) + "</div>"


def flow(steps, colour=""):
    items = []
    for i, (title, text) in enumerate(steps):
        if i:
            items.append('<div class="arrow">➜</div>')
        items.append(f'<div class="step {colour}"><b>{esc(title)}</b><p class="muted" style="margin:4px 0 0">{text}</p></div>')
    return '<div class="flow">' + "".join(items) + "</div>"


# ── static pages ────────────────────────────────────────────────────────────
HOME = wrap(f"""
<h2>Measuring sentence difficulty from eye movements, in English and Tamil</h2>
<p class="lead">When people read, their eyes stop longer on hard words. This project learns those reading times
and uses them to decide <b>which sentence is harder</b>, then makes hard Tamil sentences easier.</p>
<div class="grid g3" style="margin-top:12px">
 <div class="card blue"><span class="tag">Phase 1 · English</span>
  <h3>Replicate Singh et al. (2016)</h3>
  <p>Learn reading times from the Dundee eye-tracking corpus, then judge 117,712 Wikipedia vs Simple Wikipedia pairs.</p>
  <div class="big">74.15% · 75.26%</div><p class="muted">accuracy (paper: 73.82% · 75.21%), <b>replicated</b></p></div>
 <div class="card blue"><span class="tag">Phase 2 · English</span>
  <h3>Add GPT-2 surprisal</h3>
  <p>GPT-2 measures how unexpected each word is. Added as a feature to the reading-time model.</p>
  <div class="big">RPD 0.582 · TD 0.568</div><p class="muted">R², now <b>better than the paper</b> (0.570 · 0.516)</p></div>
 <div class="card orange"><span class="tag orange">Phase 3 · Tamil</span>
  <h3>Words or morphemes?</h3>
  <p>Built a word metric and a morphology metric, simplified hard sentences with an LLM, and tested with people,
  webcam eye-tracking and 1,904 human simplifications.</p>
  <div class="big orange">95.6% vs 93.4%</div><p class="muted">word vs morph metric on OasisSimp: <b>word-level is enough</b></p></div>
</div>
<h3>How to use this toolkit</h3>
<table class="t"><tr><th>Tab</th><th>You type</th><th>You get</th></tr>
<tr><td><b>Phase 1 · English</b></td><td>two English sentences</td><td>predicted reading time for every word, and which sentence is harder</td></tr>
<tr><td><b>Phase 2 · GPT-2</b></td><td>one English sentence</td><td>how surprising each word is, and reading times with and without GPT-2</td></tr>
<tr><td><b>Phase 3 · Tamil</b></td><td>Tamil sentences</td><td>how we built it, the 12 features of any sentence, which sentence is harder, and LLM simplification</td></tr>
</table>
""")

PHASE1_INTRO = wrap(f"""
<h2>Phase 1 · Replication of Singh et al. (2016)</h2>
<p class="lead">Type two English sentences. The toolkit runs the same two systems as the paper.</p>
{flow([("1 · Features", "19 per word: length, frequency, age of acquisition, BNC word-pair probabilities, "
                         "surprisal, parser features (see tab ②)"),
       ("2 · System 1", "ridge regression predicts the <b>reading times</b> (ms) of every word, learned from Dundee"),
       ("3 · System 2", "logistic regression compares the two lists of reading times: <b>which sentence is harder?</b>")])}
<div class="grid g2" style="margin-top:12px">
 <div class="card"><b>What the replication achieved</b>
  <table class="t" style="margin-top:6px"><tr><th>System 2 model</th><th class="c">Ours</th><th class="c">Paper</th></tr>
  <tr><td>Base reading times</td><td class="c"><b>74.15%</b></td><td class="c">73.82%</td></tr>
  <tr><td>Extended reading times</td><td class="c"><b>75.26%</b></td><td class="c">75.21%</td></tr></table>
  <p class="muted">10-fold cross-validation on 117,712 Wikipedia / Simple Wikipedia pairs.</p></div>
 <div class="card"><b>System 1, test R²</b>
  <table class="t" style="margin-top:6px"><tr><th>Measure</th><th class="c">Ours</th><th class="c">Paper</th></tr>
  <tr><td>FFD first fixation</td><td class="c">0.447</td><td class="c">0.649</td></tr>
  <tr><td>FPD first pass</td><td class="c">0.561</td><td class="c">0.600</td></tr>
  <tr><td>RPD regression path</td><td class="c">0.558</td><td class="c">0.570</td></tr>
  <tr><td>TD total duration</td><td class="c"><b>0.538</b></td><td class="c">0.516</td></tr></table>
  <p class="muted">The paper's own Table 1 and Table 2 disagree on FFD.</p></div>
</div>
<p class="muted" style="margin-top:8px"><b>① Which is harder?</b> runs System 2 exactly as in the thesis: for the
117,712 Wikipedia pairs, step 5 used the lexical features (parsing 235,000 sentences was too heavy), so the verdict
here does the same and reproduces the 75.26% model. <b>② All 19 features</b> computes every feature live,
including the thesis's left-corner parser, and shows all four System 1 reading times.</p>
""")

PHASE2_INTRO = wrap(f"""
<h2>Phase 2 · GPT-2 surprisal</h2>
<p class="lead"><b>Surprisal</b> = how unexpected a word is, given the words before it.
“I drank hot <b>coffee</b>” is expected (low surprisal); “I drank hot <b>elephant</b>” is not (high surprisal).
Readers slow down on surprising words, so GPT-2 should help predict reading time.</p>
{flow([("1 · GPT-2", "reads the sentence and gives every word a surprisal value"),
       ("2 · System 1", "predicts 4 reading times per word, with and without GPT-2"),
       ("3 · Compare", "see which words GPT-2 makes slower or faster")])}
<div class="card" style="margin-top:12px"><b>The 4 reading times</b> (all learned from the Dundee corpus)
<table class="t" style="margin-top:6px">
<tr><th>Measure</th><th>Meaning</th><th class="c">R² before</th><th class="c">R² with GPT-2</th><th class="c">Paper</th></tr>
<tr><td><b>FFD</b></td><td>first fixation: the very first look at the word</td><td class="c">0.447</td><td class="c">0.458</td><td class="c">0.649</td></tr>
<tr><td><b>FPD</b></td><td>first pass: all looks before the eyes move on</td><td class="c">0.561</td><td class="c">0.583</td><td class="c">0.600</td></tr>
<tr><td><b>RPD</b></td><td>regression path: also counts looking back</td><td class="c">0.558</td><td class="c"><b>0.582</b></td><td class="c">0.570</td></tr>
<tr><td><b>TD</b></td><td>total time on the word</td><td class="c">0.538</td><td class="c"><b>0.568</b></td><td class="c">0.516</td></tr>
</table><p class="muted">Bold = better than the paper. GPT-2 surprisal correlates with FFD at r = 0.465 (parser surprisal: −0.017).</p></div>
""")

PHASE3_STORY = wrap(f"""
<h2>Phase 3 · How we built and tested the Tamil difficulty metric</h2>
<p class="lead"><b>The question:</b> Tamil joins many parts into one word
(<span class="ta">வீட்டிலிருந்து</span> = <span class="ta">வீடு + இல் + இருந்து</span>, “from the house”).
So is a long Tamil word really hard, or should we count <b>morphemes</b> instead of words?</p>

<h3>Part A · Build the metric</h3>
<div class="grid g4">
 <div class="card orange"><span class="num">1</span><b>Collect graded text</b>
  <p>Tamil Nadu school textbooks, Class 3–12: <b>2,351</b> sentences → <b>8,004</b> pairs.<br>
  StoryWeaver books, levels 1–4: <b>7,080</b> sentences → <b>7,492</b> pairs.</p>
  <p class="muted">The grade tells us which sentence of a pair is harder.</p></div>
 <div class="card orange"><span class="num">2</span><b>Extract 12 features</b>
  <p>Each sentence goes through 4 tools: <b>Morfessor</b> (morphemes), <b>Stanza</b> (root, grammar, tree),
  <b>Tamil GPT-2</b> (surprisal), <b>wordfreq</b> (frequency).</p>
  <p class="muted">See them live in “② Analyse a sentence”.</p></div>
 <div class="card orange"><span class="num">3</span><b>Train two metrics</b>
  <p>For each pair: hard − easy features. Logistic regression learns one weight per feature.
  Score = Σ weight × feature.</p>
  <p><b>Word metric</b> = group A (4 features)<br><b>Morph metric</b> = A + B (10 features)</p></div>
 <div class="card orange"><span class="num">4</span><b>Test on unseen pairs</b>
  <table class="t"><tr><th>StoryWeaver test</th><th class="c">Acc.</th></tr>
  <tr><td>Word metric (A)</td><td class="c"><b>63.8%</b></td></tr>
  <tr><td>Morphology alone (B)</td><td class="c">60.5%</td></tr>
  <tr><td>Morph metric (A + B)</td><td class="c">63.4%</td></tr></table>
  <p class="muted">Morphology has a signal, but adds nothing.</p></div>
</div>

<h3>Part B · Use it: simplify hard sentences</h3>
<div class="grid g4">
 <div class="card blue"><span class="num blue">5</span><b>LLM simplification</b>
  <p><b>60</b> hard sentences (textbook Class 11–12, StoryWeaver level 4) rewritten for children by an LLM.</p>
  <p>Words <b>12.8 → 6.8</b> · morphemes/word 1.44 → 1.34 · non-finite verbs 2.5 → 1.0</p>
  <p class="muted">Both metrics: <b>100%</b> of rewrites easier.</p></div>
</div>

<h3>Part C · Check it with real readers</h3>
<div class="grid g4">
 <div class="card green"><span class="num" style="background:#1E7B45">6</span><b>Human rating study</b>
  <p><b>12</b> native readers rated the 60 pairs (Google Form).</p>
  <p>Difficulty <b>2.31 → 1.97</b> (1 = very easy, 5 = very hard)</p>
  <p class="muted">Agreement with readers: word ρ = 0.32, morph ρ = 0.28.</p></div>
 <div class="card green"><span class="num" style="background:#1E7B45">7</span><b>Own webcam eye-tracking</b>
  <p>No Tamil eye-tracking corpus exists, so I built a web app (WebGazer). <b>100</b> sentences, <b>5</b> readers,
  1,021 words with FFD / FPD / RPD / TD.</p>
  <p>Total reading time R²: word <b>0.103</b>, morph alone 0.071, both 0.099</p></div>
 <div class="card green"><span class="num" style="background:#1E7B45">8</span><b>External test: OasisSimp</b>
  <p><b>1,904</b> Tamil sentences simplified by trained people, never used for training.</p>
  <p>Picks the original as harder: word <b>95.6%</b>, morph 93.4%</p>
  <p class="muted">Same-length pairs: 90.5% vs 83.3%. Morph is significantly worse.</p></div>
</div>

<div class="verdict orange" style="margin-top:16px">Finding: morphology carries difficulty on its own,
but it adds nothing beyond word-level features, in all four tests.</div>
<p class="muted">Why? Long Tamil words usually have more morphemes, so the two kinds of features measure almost the
same thing, and the Tamil tools are still noisy (you can see Morfessor and Stanza errors in “② Analyse a sentence”).
Same idea as the English work: steps 1–4 do the job of <b>System 2</b>, step 7 is <b>System 1</b> on Tamil reading times.</p>
""")

ANALYSE_INTRO = wrap(f"""
<h3>② Analyse one Tamil sentence: the 12 features, live</h3>
{flow([("1 · Your sentence", "type or paste any Tamil sentence"),
       ("2 · Four tools", "Morfessor splits words, Stanza finds roots and grammar, Tamil GPT-2 gives surprisal, wordfreq gives frequency"),
       ("3 · 12 features", "averaged over the words of the sentence"),
       ("4 · Score", "Σ weight × feature → which StoryWeaver level it reads like")], "orange")}
<table class="t" style="margin-top:10px">
<tr><th>Group</th><th>Features</th></tr>
<tr><td><b>A · word-level</b> (like English)</td><td>number of words · letters per word · word frequency · GPT-2 surprisal</td></tr>
<tr><td><b>B · morphology</b> (new for Tamil)</td><td>morphemes per word · stem surprisal · suffix surprisal · grammar marks per word · root frequency · non-finite verbs</td></tr>
<tr><td><b>C · syntax</b></td><td>tree depth · word-to-head distance</td></tr>
</table>
""")
