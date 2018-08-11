from cnt_rulebase import (
    sentseg,
    dlmseg,
    replace_english_chars,
    replace_digits,
)

from .const import TOKEN_DLM, TOKEN_EN, TOKEN_NUM


def select_first(seq):
    return [p[0] for p in seq]


def break_to_sentences(text):
    return select_first(sentseg(text, enable_comma=True))


def break_to_segments(text):
    return dlmseg(text)


def preprocess_segments(segs, gap=0):
    if not segs:
        return []

    processed = []
    pre_end, _ = segs[0][1]
    pre_end -= 1

    for text, (start, end) in segs:
        if start - pre_end > gap + 1:
            processed.append(TOKEN_DLM)
        pre_end = end

        text = replace_english_chars(text, TOKEN_EN)
        text = replace_digits(text, TOKEN_NUM)
        processed.append(text)

    return processed


