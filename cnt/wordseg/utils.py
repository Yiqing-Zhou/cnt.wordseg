from cnt.rulebase import (
    sentseg,
    dlmseg,
    replace_english_chars,
    replace_digits,
)

from .const import (
    TOKEN_DLM, TOKEN_EN, TOKEN_NUM, TOKEN_BMES_BREAK,
)


def select_first(seq):
    return [p[0] for p in seq]


def break_to_sentences(text):
    return select_first(sentseg(text, enable_comma=True))


def break_to_segments(text):
    return dlmseg(text)


def preprocess_segments(segs, gap=0):
    if not segs:
        return []

    text = ' '.join(select_first(segs))
    text = replace_english_chars(text, TOKEN_EN)
    text = replace_digits(text, TOKEN_NUM)
    processed_texts = text.split(' ')
    assert len(processed_texts) == len(segs)

    processed = []
    pre_end, _ = segs[0][1]
    pre_end -= 1

    for text, (_, (start, end)) in zip(processed_texts, segs):
        if start - pre_end > gap + 1:
            processed.append(TOKEN_DLM)
        pre_end = end

        processed.append(text)

    return processed


def extract_chars(text):
    ret = []
    idx = 0
    while idx < len(text):
        space_mode = False
        while text[idx].isspace():
            space_mode = True
            idx += 1
        if space_mode:
            ret.append(TOKEN_DLM)
            continue

        if text[idx] != '<':
            ret.append(text[idx])
            idx += 1

        else:
            end = idx + 1
            while end < len(text) and text[end] != '>':
                end += 1
            assert end < len(text)
            ret.append(text[idx:end + 1])
            idx = end + 1
    return ret


def split_bmes_lines(lines, tok_prefix=None, tok_suffix=None):
    groups = []

    begin = 0
    while begin < len(lines):
        group = []
        if tok_prefix:
            group.append(tok_prefix)

        end = begin
        while end < len(lines) and lines[end] != TOKEN_BMES_BREAK:
            group.append(lines[end])
            end += 1

        if tok_suffix:
            group.append(tok_suffix)

        groups.append(group)
        begin = end + 1

    return groups
