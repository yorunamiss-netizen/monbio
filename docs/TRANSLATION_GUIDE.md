# Translation guide

Three steps: machine translation once, post-editing in batches, import. Budget: about 5 hours a week.

## Step 1: machine translation (once, about 30 minutes)

The raw machine translation is frozen once and never edited. It is its own study condition (`mn_mt`),
and edit rates are measured against it.

1. Open Google Sheets, create a blank sheet, then **File > Import > Upload** `data/mt/mt_input.csv`.
   Choose "Replace spreadsheet", separator "Comma", and **untick** "Convert text to numbers, dates, and
   formulas" (otherwise an option like `-1` turns into a number).
2. Click cell **C2** and type:

   ```
   =GOOGLETRANSLATE(B2,"en","mn")
   ```

   If Sheets shows a formula parse error, your locale uses semicolons: `=GOOGLETRANSLATE(B2;"en";"mn")`.
3. Double-click the small square at the bottom-right corner of C2. The formula fills down to the last row.
4. Wait until no cell says `Loading...`. With 1,950 rows this can take a few minutes. Scroll to the bottom to check.
5. Select column C, copy, then **Edit > Paste special > Values only**. This freezes the translation, so it
   cannot change later.
6. **File > Download > Comma-separated values (.csv)**. Then run:

   ```bash
   python scripts/import_mt.py --csv path/to/downloaded.csv --date 2026-11-24
   ```

   Use the date you did the translation. If the script lists problems (an error value, an empty cell,
   an untranslated word), fix those rows in the sheet (type the formula again in that row, wait, paste
   values), download again, and re-run.
7. Make the post-editing sheets:

   ```bash
   python scripts/make_translation_batches.py
   ```

   This writes `data/translation/batch_01.xlsx` to `batch_08.xlsx`. Existing files are never overwritten.

## Step 2: post-editing (batches of 50 items)

Work through the batches in order: batch 1 first (it is the pilot set). You can edit in Excel,
LibreOffice, or Google Sheets (upload the .xlsx, edit, then **File > Download > Microsoft Excel** and
replace the file in `data/translation/`).

**Drafts from the Google Sheet (first pass, 2026-11-20 to 21).** The first pass was done in the Google
Sheet `bio` (columns `mn_draft`, `note`, `edit_type`). It is saved in `data/drafts/drafts_2026-11-21.jsonl`,
and the workbooks start from it: column E holds your draft, column H your note, rows with CHECK in the note
have the flag `term`, and column I (`auto_check`) has automatic hints. What is left is the **final check**:
read each row, fix it if needed, type `x` in column F. From now on edit the workbooks, not the Google Sheet;
the import reads only the workbooks.

To rebuild the workbooks from a newer state of the Google Sheet (only before you start checking:
`--overwrite` replaces checked work too): download the sheet as CSV, then

```bash
python scripts/import_drafts.py --csv path/to/sheet.csv
```

```bash
python scripts/make_translation_batches.py --drafts data/drafts/drafts_YYYY-MM-DD.jsonl --overwrite
```

**Rules**

1. **Edit only column E (`mn_final`).** It starts as your draft, or as the machine translation if the
   workbooks were made without drafts. Never change
   column D (`mt_raw`); the import stops if it was changed.
2. **Do not look up the answers while editing.** The sheet has no answer key and no option letters on
   purpose. Give all four options the same care. The import measures whether correct options were
   edited differently from wrong ones.
3. **Keep the meaning, not the English word order.** Write natural Mongolian a biology teacher would use.
4. **Do not make an option more or less specific, longer, or more "correct-sounding"** than the English.
5. **Keep as in English:** numbers, units, chemical formulas, gene and protein names (ATP, BRCA1, pH, DNA),
   and Latin species names (*Homo sapiens*).
6. **Terms:** use standard Mongolian school or university biology terms. Record every term decision in
   `data/glossary.csv` (columns `mongolian`, `source`, `note`) and reuse it, so a term is always
   translated the same way.
7. **Mark a row done:** type `x` in column F (`checked`). An item counts as verified only when all 5 of
   its rows (question and 4 options) are checked.
8. **Flags (column G):** `term` if a term could not be verified against a Mongolian source (these items
   get a separate analysis; if you find a source, write it in column H and clear the flag). `unsure` if
   you want a second look for another reason. `drop` if the item cannot be translated sensibly (for
   example it depends on an English word play); explain why in column H.
9. **Log your time** in the `log` sheet: date, minutes, items done. This gives the cost per item for the report.

## Step 3: import

```bash
python scripts/import_translations.py
```

It writes `data/items_mn.jsonl` (verified items only), `data/mn_segments.jsonl`, and
`data/translation_report.md` with:

- verified items per batch, minutes per item;
- how much you changed the machine translation (character and word edit rates);
- the blind-editing check: edit rate and length change on correct options minus distractors. Values close
  to 0 mean the answer did not leak into the editing.

Run it whenever you like; it only reads the sheets.
