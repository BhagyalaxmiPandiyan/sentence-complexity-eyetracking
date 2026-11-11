# How to run the native-speaker study (Phase C)

Built by `src/step8_score_simplified.py`:

- `make_google_form.gs`: a script that creates the Google Form(s) for you
- `items_key.csv`: the hidden answer key (which side is the original). **Never share it.**

Each form has 30 items. Each item shows sentence A and sentence B (original and
simplified, in random order) and asks 5 quick questions: which is easier, the
difficulty of A (1-5), the difficulty of B (1-5), whether the meaning is the same (1-5),
and which has mistakes. It takes about 20 minutes.

## 1. Create the forms (5 min, free)

1. Go to **script.google.com**, sign in with your Google account, and click **New project**.
2. Delete the sample code. Open `make_google_form.gs` in Notepad, copy all of it, and paste it in.
3. Click **Save**, then **Run** (the function `makeForms`).
4. Google asks for permission. Click Review permissions, choose your account, then
   Advanced → Go to project → Allow.
5. Open **Execution log**: there is one `share:` link per form. The forms are also in your Google Drive.
6. Open each form once and check that the Tamil text looks right.

## 2. Collect ratings

- Aim for **10-20 native Tamil readers** (friends, family, classmates, teachers).
- Split them evenly across the forms (with 2 forms, half get Form 1 and half get Form 2),
  so that each item gets **at least 5 raters**.
- Message to send: *"I need 20 minutes for my thesis: read pairs of Tamil sentences
  and say which is easier. No right or wrong answers."*
- Don't tell raters which sentence is the simplified one.

## 3. Download and analyse

1. In each form go to **Responses → ⋮ → Download responses (.csv)**.
2. Unzip, and rename the files to `form1.csv`, `form2.csv`, ... (matching the form number in its title).
3. Put them in `Dataset/human_study/responses/`.
4. Run `python src/step9_human_eval.py` → `results/human_eval_summary.md`.

Ethics: the form collects only initials and reading level. Tell raters what the data
is for, and ask your supervisor whether your university needs an ethics form for an
anonymous rating study.
