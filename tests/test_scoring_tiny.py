"""End-to-end tests of scoring with a tiny random model built locally (no downloads).

The model's answers are meaningless; the tests check the plumbing: letter tokens, chat template,
left padding, rotation bookkeeping, the score.py CLI and the pilot report.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from monbio.prompts import build_prompts  # noqa: E402
from monbio.scoring import (  # noqa: E402
    Scorer,
    find_decision_step,
    is_standalone_letter,
    letter_token_ids,
    render_chat,
)
from monbio.utils import read_jsonl, write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CORPUS = [
    "The following is a multiple choice question. Reply with only the letter (A, B, C, or D).",
    "Дараах олон сонголттой асуултад хариулна уу. Зөвхөн зөв хариултын үсгийг бичнэ үү.",
    "Which organelle makes ATP? Mitochondrion Ribosome Lysosome Golgi apparatus",
    "Аль органелл ATP үүсгэдэг вэ? Митохондри Рибосом Лизосом Гольджийн аппарат",
    "A. B. C. D. user: assistant:",
] * 20

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


def make_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<s>", "</s>", "<pad>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(CORPUS, trainer)
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    hf.chat_template = CHAT_TEMPLATE
    return hf


def make_model(tokenizer):
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tiny():
    tok = make_tokenizer()
    return make_model(tok), tok


ITEMS_EN = [
    {"item_id": "cb-001", "subject": "college_biology", "question": "Which organelle makes ATP?",
     "options": ["Ribosome", "Mitochondrion", "Golgi apparatus", "Lysosome"], "answer": 1,
     "order": 0, "batch": 1, "pilot": True},
    {"item_id": "an-002", "subject": "anatomy", "question": "Which organelle makes ATP in muscle cells during exercise?",
     "options": ["Lysosome", "Ribosome", "Mitochondrion", "Golgi apparatus"], "answer": 2,
     "order": 1, "batch": 1, "pilot": True},
]
ITEMS_MN = [
    {**ITEMS_EN[0], "question": "Аль органелл ATP үүсгэдэг вэ?", "options": ["Рибосом", "Митохондри", "Гольджийн аппарат", "Лизосом"]},
    {**ITEMS_EN[1], "question": "Аль органелл ATP үүсгэдэг вэ?", "options": ["Лизосом", "Рибосом", "Митохондри", "Гольджийн аппарат"]},
]


def test_letter_tokens_are_distinct(tiny):
    _, tok = tiny
    ids = letter_token_ids(tok)
    assert list(ids) == ["A", "B", "C", "D"]
    flat = [t for v in ids.values() for t in v]
    assert len(flat) == len(set(flat))


class FakeTokenizer:
    """Just enough of a tokenizer for letter_token_ids: a vocabulary and single-token encoding."""

    def __init__(self, tokens):
        self.vocab = {t: i for i, t in enumerate(tokens)}

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [-1, -1]


def test_letter_tokens_include_merged_forms():
    # Qwen3, Qwen3.5 and GLM-4 encode "(C)" as ["(C", ")"], so "(C" has to count as C.
    tok = FakeTokenizer(["A", "B", "C", "D", "ĠA", "ĠB", "ĠC", "ĠD", "(C", "Ġ(C", "[A", "**B", "▁(B", '"D',
                         "(Cell", "(", "'d", "AB", "(1"])
    v = tok.vocab
    ids = letter_token_ids(tok)
    assert ids["C"] == sorted([v["C"], v["ĠC"], v["(C"], v["Ġ(C"]])
    assert v["[A"] in ids["A"] and v["**B"] in ids["B"] and v["▁(B"] in ids["B"] and v['"D'] in ids["D"]
    every = {t for t_ids in ids.values() for t in t_ids}
    assert not {v["(Cell"], v["("], v["'d"], v["AB"], v["(1"]} & every


def test_letters_inside_words_are_not_answers(tiny):
    _, tok = tiny
    owner = {t: k for k, v in letter_token_ids(tok).items() for t in v}

    def detected(text):
        toks = tok.encode(text, add_special_tokens=False)
        step = find_decision_step(toks, set(owner), set(), standalone=lambda t: is_standalone_letter(tok, toks, t))
        return owner[toks[step]] if step >= 0 else None

    assert detected("C") == "C"
    assert detected("The answer is: **C**") == "C"
    assert detected("(C) Mitochondrion") == "C"
    assert detected("Зөв хариулт: C") == "C"
    assert detected("DNA") is None  # the D of a word is not the answer D
    assert detected("CD4") is None
    assert detected("Drosophila cells, so B.") == "B"


def test_safetensors_pread_reads_and_restores(tmp_path):
    from safetensors.torch import save_file
    import transformers.modeling_utils as modeling_utils

    from monbio.scoring import safetensors_pread

    path = tmp_path / "w.safetensors"
    save_file({"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, str(path))
    original = modeling_utils.safe_open
    with safetensors_pread():
        assert modeling_utils.safe_open is not original
        with modeling_utils.safe_open(str(path), framework="pt", device="cpu", backend="mmap") as f:
            assert torch.equal(f.get_tensor("w"), torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert modeling_utils.safe_open is original


def test_render_chat_modes(tiny):
    _, tok = tiny
    text, mode = render_chat(tok, "hello")
    assert mode == "chat" and text.endswith("assistant:") and text.startswith("<s>")
    plain = make_tokenizer()
    plain.chat_template = None
    text, mode = render_chat(plain, "hello\nAnswer:")
    assert mode == "plain" and text == "hello\nAnswer:"


def test_find_decision_step():
    letters, stop = {10, 11, 12, 13}, {0}
    assert find_decision_step([5, 6, 12, 7], letters, stop) == 2  # "The answer is C"
    assert find_decision_step([11], letters, stop) == 0  # letter first
    assert find_decision_step([5, 0, 12], letters, stop) == -1  # stopped before any letter
    assert find_decision_step([5, 6, 7], letters, stop) == -1  # budget used up
    # A letter that is part of a word is skipped; the next standalone letter counts.
    assert find_decision_step([13, 5, 11], letters, stop, standalone=lambda t: t != 0) == 2


def test_batching_does_not_change_scores(tiny):
    model, tok = tiny
    prompts = build_prompts({"en": ITEMS_EN, "mn_mt": ITEMS_MN}, ["instr_en", "instr_mn"], rotations=4)
    texts = [p.text for p in prompts]
    one = Scorer(model, tok, batch_size=1, max_new_tokens=4).score_texts(texts)
    many = Scorer(model, tok, batch_size=7, max_new_tokens=4).score_texts(texts)
    a = np.array([r["first_lp"] for r in one])
    b = np.array([r["first_lp"] for r in many])
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()
    # Where both runs took the same greedy path, the decision readouts agree too.
    for x, y in zip(one, many):
        if x["answer_step"] == y["answer_step"]:
            assert np.allclose(x["lp"], y["lp"], atol=1e-4)
    assert not any(r["nonfinite"] for r in many)
    assert all(0 < r["first_letter_mass"] <= 1 for r in many)
    assert all(r["no_letter"] == (r["answer_step"] < 0) for r in many)
    # Mongolian prompts are longer in tokens for this tokenizer, as for real ones.
    assert len({r["n_tokens"] for r in many}) > 1


def test_batch_consistency_check(tiny):
    from monbio.scoring import batch_consistency

    model, tok = tiny
    prompts = build_prompts({"en": ITEMS_EN, "mn_mt": ITEMS_MN}, ["instr_en", "instr_mn"], rotations=1)
    scorer = Scorer(model, tok, batch_size=4, max_new_tokens=2)
    assert batch_consistency(scorer, [p.text for p in prompts]) < 1e-4
    assert scorer.batch_size == 4  # restored after the check


def test_score_prompts_rows(tiny):
    model, tok = tiny
    prompts = build_prompts({"en": ITEMS_EN[:1]}, ["instr_en"], rotations=4)
    rows = Scorer(model, tok, batch_size=4).score_prompts(prompts, "tiny")
    assert [r["perm"] for r in rows] == ["0123", "1230", "2301", "3012"]
    assert set(rows[0]) >= {"model", "condition", "variant", "item_id", "rotation", "lp_A", "lp_D", "letter_mass",
                            "first_lp_A", "first_letter_mass", "answer_step", "no_letter", "preamble"}


def test_cli_end_to_end(tiny, tmp_path):
    model, tok = tiny
    model_dir = tmp_path / "tiny-model"
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    data = tmp_path / "data"
    write_jsonl(data / "items_en.jsonl", ITEMS_EN)
    write_jsonl(data / "items_mn_mt.jsonl", ITEMS_MN)
    out = tmp_path / "results"

    cmd = [
        sys.executable, str(ROOT / "scripts" / "score.py"), "--model", str(model_dir), "--items", "pilot",
        "--conditions", "en,mn_mt", "--variants", "instr_en,instr_mn",
        "--condition-file", f"en={data / 'items_en.jsonl'}", "--condition-file", f"mn_mt={data / 'items_mn_mt.jsonl'}",
        "--dtype", "float32", "--device-map", "none", "--batch-size", "5", "--out", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    csvs = list(out.glob("*.csv"))
    assert len(csvs) == 1 and not list(out.glob("*.error.txt")), result.stdout
    meta = list(out.glob("*.meta.json"))
    assert len(meta) == 1

    report = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pilot_report.py"), "--results", str(out), "--items", str(data / "items_en.jsonl")],
        capture_output=True, text=True,
    )
    assert report.returncode == 0, report.stdout + report.stderr
    summary = (out / "summary.csv").read_text()
    assert "mn_mt" in summary and "instr_mn" in summary
    items = (out / "item_scores.csv").read_text().splitlines()
    assert len(items) == 1 + 2 * 3  # 2 items x (en/instr_en, mn_mt/instr_en, mn_mt/instr_mn)
    assert "Decisions" in (out / "report.md").read_text()

    # Running again without --overwrite leaves the result alone.
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert "exists, skipped" in again.stdout
