import re

from monbio.prompts import LETTERS, TEMPLATES, allowed_variants, build_prompts, format_question, rotation_perm

# Cyrillic capitals that look like Latin A, B, C, D-ish letters.
CYRILLIC_LOOKALIKES = re.compile(r"[АВСЕНКМОРТХ]")


def test_rotation_perm():
    assert rotation_perm(0) == (0, 1, 2, 3)
    assert rotation_perm(1) == (1, 2, 3, 0)
    assert rotation_perm(3) == (3, 0, 1, 2)
    # Over all rotations every option appears once at every position.
    for pos in range(4):
        assert sorted(rotation_perm(r)[pos] for r in range(4)) == [0, 1, 2, 3]


def option_lines(text):
    return [line for line in text.splitlines() if re.match(r"^\w\. ", line)]


def test_format_question_rotates_options():
    text, perm = format_question("Q?", ["w", "x", "y", "z"], "instr_en", rotation=1)
    assert perm == (1, 2, 3, 0)
    assert option_lines(text) == ["A. x", "B. y", "C. z", "D. w"]
    # The instruction and the answer cue come after the options.
    assert text.splitlines()[-1] == "Answer:"


def test_labels_are_ascii_in_every_variant():
    for variant, template in TEMPLATES.items():
        text, _ = format_question("Асуулт?", ["нэг", "хоёр", "гурав", "дөрөв"], variant)
        labels = [line[0] for line in option_lines(text)]
        assert labels == list(LETTERS)
        assert all(ch.isascii() for ch in labels)
        # The instruction names the letters; they must be ASCII too.
        for letter in LETTERS:
            assert letter in template["instruction"], (variant, letter)
        for part in template.values():
            standalone = re.findall(r"(?<!\w)\w(?!\w)", part)
            assert not [ch for ch in standalone if CYRILLIC_LOOKALIKES.match(ch)], variant


def test_allowed_variants_and_prompt_count():
    assert allowed_variants("en") == ("instr_en",)
    assert allowed_variants("mn_mt") == ("instr_en", "instr_mn")
    item = {"item_id": "cb-001", "question": "Q?", "options": ["w", "x", "y", "z"]}
    prompts = build_prompts({"en": [item], "mn_mt": [item]}, ["instr_en", "instr_mn"], rotations=4)
    assert len(prompts) == 4 + 8
    assert {(p.condition, p.variant) for p in prompts} == {("en", "instr_en"), ("mn_mt", "instr_en"), ("mn_mt", "instr_mn")}
