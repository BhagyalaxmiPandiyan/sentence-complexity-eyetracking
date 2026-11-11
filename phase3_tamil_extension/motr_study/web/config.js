// Study settings. The only line you MUST change before sharing is ENDPOINT.
const CONFIG = {
  // Google Apps Script web-app URL (see HOW_TO_RUN.md, step 2).
  // Leave empty for test mode: nothing is uploaded, data is downloaded as a file at the end.
  ENDPOINT: "",

  BREAK_AFTER: 50,     // rest screen after this many main sentences
  FOCUS_RX: 70,        // clear (unblurred) area around the mouse: half-width in px
  FOCUS_RY: 32,        //                                         half-height in px
  BLUR_PX: 4,          // blur strength of the rest of the sentence
  MIN_TRIAL_MS: 800,   // ignore Space presses faster than this
  MIN_PROBE_ACC: 0.7,  // shown to the researcher only (analysis uses the same cut-off)
};
