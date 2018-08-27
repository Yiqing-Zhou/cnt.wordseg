from cnt.wordseg.const import (
    generate_token,
)


def test_generate_token():
    assert '<a>' == generate_token('a')
    assert '</a>' == generate_token('a', True)
