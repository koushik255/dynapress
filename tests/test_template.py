"""Template matching: the mechanism that stops the model from rewriting stored text."""

from neural_compression.models import Gap, advance

T = [" progr", Gap(4, 5), " disco", Gap(4, 5), ".\n"]
END = (len(T), 0)


def run(text, template=T):
    return advance(template, (0, 0), text)


def test_accepts_valid_completions():
    assert run(" programmer discovered.\n") == END
    assert run(" programs discoverer.\n") is None  # "programs": gap of 3 < min 4
    assert run(" programme discovery.\n") == END


def test_rejects_changed_prefix_or_separator():
    assert run(" Programmer discovered.\n") is None
    assert run(" programmer, discovered.\n") is None
    assert run(" programmer discovered!\n") is None


def test_gap_length_bounds():
    assert run(" program discovered.\n") is None  # gap "am" shorter than 4
    assert run(" programmers discovered.\n") is None  # gap "ammers" longer than 5
    assert run(" progr4mmer discovered.\n") == END  # digits are word characters (\w)


def test_gap_only_word_characters():
    assert run(" progr-mmer discovered.\n") is None


def test_incremental_matches_whole():
    text = " programmer discovered.\n"
    state = (0, 0)
    for piece in [" pro", "gram", "mer dis", "covered", ".\n"]:
        state = advance(T, state, piece)
        assert state is not None
    assert state == run(text) == END


def test_zero_length_gap_allowed():
    template = [" th", Gap(0, 2), " end\n"]
    assert advance(template, (0, 0), " th end\n") == (3, 0)
    assert advance(template, (0, 0), " the end\n") == (3, 0)
    assert advance(template, (0, 0), " there end\n") is None


def test_text_beyond_template_rejected():
    assert run(" programmer discovered.\nmore") is None
