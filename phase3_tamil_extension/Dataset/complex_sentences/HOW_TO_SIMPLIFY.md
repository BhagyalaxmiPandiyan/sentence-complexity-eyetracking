# How to get LLM simplifications for free (Phase B)

`original.csv` holds the 60 complex sentences (30 textbook Class 11-12, 30 StoryWeaver
level 4), picked at random by `src/step7_select_complex.py`.

One system is already done: `simplified_claude-opus-5-5.txt` (Claude Opus 5.5, run
inside Claude Code on 2026-10-13 with exactly the prompt in `prompt_batch*.txt`).

## Add a second system from a free web chat (optional, ~15 min)

A second LLM makes the thesis stronger ("two LLMs, same prompt").

1. Open a free chat, e.g. **Gemini** (gemini.google.com) or **ChatGPT** (chatgpt.com).
   Start a **new chat**.
2. Open `prompt_batch1.txt`, copy everything, paste it into the chat, and send.
3. Copy the reply. It should be 20 lines like `S01 | ...`.
4. Repeat steps 2-3 for `prompt_batch2.txt` and `prompt_batch3.txt` in the same chat.
5. Paste all three replies into one new file in this folder, named after the model,
   e.g. `simplified_gemini.txt` or `simplified_chatgpt.txt`. Save it as UTF-8.
6. Write down the model name and the date (for the thesis methods section).
7. Run `python src/step8_score_simplified.py`. It warns you if any ID is missing.

Extra text from the chat (like "Here are the sentences:") is ignored. Only lines
starting with an ID count.

**Important:** step 8 rebuilds the Google Form each time it runs. Add every system
**before** you create the form and send it to raters. Do not rerun step 8 after
responses start coming in.
