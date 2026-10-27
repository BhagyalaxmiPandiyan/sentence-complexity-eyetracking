// Study settings. The only line you MUST change before sharing is ENDPOINT.
const CONFIG = {
  // Google Apps Script web-app URL (HOW_TO_SHARE.md, step 2).
  // Empty = test mode: nothing is uploaded; the data is downloaded as a
  // file at the end instead.
  ENDPOINT: "",

  // Display
  FONT_PX: 30,          // largest font tried; shrinks if a sentence does not fit
  MIN_FONT_PX: 20,
  WORD_GAP_PX: 40,      // space between words (wider = easier word assignment)
  LINE_SPACING: 3.0,    // line pitch as a multiple of the text height
  MARGIN_PX: 80,

  // Session
  BLOCK_SIZE: 25,       // main sentences between breaks (100 -> 4 parts)
  CAL_CLICKS: 5,        // clicks per calibration dot at the start
  RECAL_CLICKS: 3,      // clicks per dot when recalibrating at a break
  MAX_ERROR_PX: 150,    // validation error above this asks to recalibrate
  MIN_TRIAL_MS: 800,    // ENTER faster than this is ignored
  DRIFT_WINDOW_MS: 500, // gaze before SPACE on the dot used for drift correction
  MIN_DRIFT_SAMPLES: 5,

  // Fixations and noise filter (same as the desktop tool)
  DISPERSION_PX: 100,
  MIN_FIX_MS: 80,
  MAX_FIX_MS: 1200,
  MAX_GAP_MS: 100,
  LINE_BAND_PX: 150,
  MIN_BLINK_MS: 60,
  MIN_VALID: 0.7,
  MAX_OFF_TEXT: 0.5,
};
