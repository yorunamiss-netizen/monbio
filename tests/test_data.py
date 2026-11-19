from collections import Counter

from monbio.data import (
    STRICT_POSITIONAL,
    Item,
    assign_batches,
    glossary_candidates,
    review_flags,
    stratified_order,
    structural_problems,
)


def make(item_id="cb-001", subject="college_biology", options=None, error="ok", answer=0, question="Q?"):
    return Item(item_id, subject, 1, question, options or ["w", "x", "y", "z"], answer, error)


def test_strict_positional_patterns():
    for text in ["All of the above", "none of these", "Both of the above", "neither of the preceding", "all the above"]:
        assert STRICT_POSITIONAL.search(text), text
    for text in ["All cells divide", "None", "Both mitosis and meiosis", "These are prokaryotes"]:
        assert not STRICT_POSITIONAL.search(text), text


def test_structural_problems():
    assert structural_problems(make()) == []
    assert "redux:wrong_groundtruth" in structural_problems(make(error="wrong_groundtruth"))
    assert "duplicate_options" in structural_problems(make(options=["a", "A", "b", "c"]))
    assert "self_referential_option" in structural_problems(make(options=["a", "b", "c", "All of the above"]))
    assert "n_options=3" in structural_problems(make(options=["a", "b", "c"]))


def test_review_flags():
    assert "letter_combination" in review_flags(make(options=["a", "b", "c", "A and B"]))
    assert "parenthesized_letter" in review_flags(make(options=["(a) only", "b", "c", "d"]))
    assert "figure_reference" in review_flags(make(question="In the figure below, what is X?"))
    assert review_flags(make(options=["Mitosis and meiosis", "DNA", "RNA", "ATP"])) == []


def _items():
    sizes = {"college_biology": 90, "high_school_biology": 95, "medical_genetics": 80, "anatomy": 85}
    items = []
    for subject, n in sizes.items():
        for i in range(n):
            items.append(make(item_id=f"{subject[:2]}-{i:03d}", subject=subject))
    return items, sizes


def test_stratified_order_is_deterministic_and_balanced():
    items, sizes = _items()
    a = [it.item_id for it in stratified_order(items, seed=1)]
    b = [it.item_id for it in stratified_order(items, seed=1)]
    c = [it.item_id for it in stratified_order(items, seed=2)]
    assert a == b and a != c
    assert sorted(a) == sorted(it.item_id for it in items)

    ordered = stratified_order(items, seed=1)
    total = len(ordered)
    counts = Counter()
    for k, item in enumerate(ordered, start=1):
        counts[item.subject] += 1
        for subject, n in sizes.items():
            assert abs(counts[subject] - k * n / total) <= 2, (k, subject)


def test_assign_batches():
    items, _ = _items()
    ordered = stratified_order(items, seed=3)
    assign_batches(ordered, batch_size=50, pilot_batches=1)
    assert [it.order for it in ordered] == list(range(len(ordered)))
    assert ordered[0].batch == 1 and ordered[49].batch == 1 and ordered[50].batch == 2
    assert sum(it.pilot for it in ordered) == 50


def test_glossary_candidates():
    items = [
        make(item_id="a", question="Which enzyme unwinds DNA during replication?", options=["helicase", "ligase", "primase", "DNA polymerase"]),
        make(item_id="b", question="DNA replication is semiconservative because", options=["helicase", "x1", "x2", "x3"]),
    ]
    terms = {row["term"]: row["n_items"] for row in glossary_candidates(items)}
    assert terms["replication"] == 2
    assert terms["helicase"] == 2
    assert "which" not in terms
