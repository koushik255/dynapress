"""Experiment contracts: book isolation and answer-only supervision."""

import json
import random
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from neural_compression.finetune import (
    collate,
    encode_rows,
    evaluate,
    prepare,
    read_rows,
    training_example,
)


@pytest.fixture
def tokenizer():
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    backend = Tokenizer(BPE(vocab, []))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>"
    )


def make_manifest(tmp_path):
    shared = "Everyone gathered around the ancient fireplace."
    books = []
    for split, word in (
        ("train", "Amber"),
        ("validation", "Bronze"),
        ("test", "Copper"),
    ):
        path = tmp_path / f"{split}.txt"
        text = f"{word} lanterns illuminated the narrow passage.\n\n{shared}"
        path.write_text(
            "Header not in the book.\n*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            + text
            + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nFooter not in the book."
        )
        books.append({"id": split, "split": split, "path": path.name})
    manifest = tmp_path / "books.json"
    manifest.write_text(json.dumps({"books": books}))
    return manifest


def test_prepare_keeps_books_disjoint_and_removes_shared_sentences(tmp_path):
    manifest = make_manifest(tmp_path)
    out = tmp_path / "prepared"
    first = prepare(manifest, out, per_book=10)
    assert prepare(manifest, out, per_book=10) == first
    for split, word in (
        ("train", "Amber"),
        ("validation", "Bronze"),
        ("test", "Copper"),
    ):
        assert read_rows(out / f"{split}.jsonl") == [
            {
                "book_id": split,
                "text": f"{word} lanterns illuminated the narrow passage.",
            }
        ]


@pytest.mark.parametrize("duplicate_id", [False, True])
def test_prepare_rejects_same_book_across_splits(tmp_path, duplicate_id):
    manifest = make_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    if duplicate_id:
        data["books"][1]["id"] = data["books"][0]["id"]
    else:
        data["books"][1]["path"] = data["books"][0]["path"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate book"):
        prepare(manifest, tmp_path / "prepared")


@pytest.mark.parametrize("mixed", [False, True])
def test_supervision_excludes_prompt_and_padding(tokenizer, mixed):
    text = '"Ravenscroft counted 42 unusually bright lanterns."'
    example = training_example(text, tokenizer, random.Random(3), mixed)
    target = [i for i in example["labels"] if i != -100]
    assert (
        tokenizer.decode(target)
        == ' Ravenscroft counted 42 unusually bright lanterns."\n'
    )
    boundary = len(example["input_ids"]) - len(target)
    assert example["labels"][:boundary] == [-100] * boundary
    prompt = tokenizer.decode(example["input_ids"][:boundary])
    assert prompt.endswith("\nOriginal:")
    assert "42" in prompt.rsplit("Compressed:", 1)[1]
    short = training_example(
        "Five small birds flew swiftly.", tokenizer, random.Random(7), mixed
    )
    batch = collate([example, short], tokenizer.pad_token_id)
    width = max(len(e["input_ids"]) for e in (example, short))
    for i, e in enumerate((example, short)):
        n = len(e["input_ids"])
        assert batch["labels"][i, :n].tolist() == e["labels"]
        assert batch["labels"][i, n:].tolist() == [-100] * (width - n)
        assert batch["attention_mask"][i, :n].tolist() == [1] * n
        assert batch["attention_mask"][i, n:].sum() == 0


def test_oversized_examples_are_rejected_not_truncated(tokenizer):
    with pytest.raises(ValueError, match="max_length"):
        encode_rows(
            [{"text": "Five small birds flew swiftly."}], tokenizer, 0, max_length=10
        )


def test_prepare_rejects_changed_source(tmp_path):
    manifest = make_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["books"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source checksum mismatch"):
        prepare(manifest, tmp_path / "prepared")


def test_test_evaluation_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="held out"):
        evaluate(SimpleNamespace(split="test", final_test=False))
