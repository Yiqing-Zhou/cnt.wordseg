def generate_token(tag, add_slash=False):
    return f'<{"/" if add_slash else ""}{tag}>'


TOKEN_DLM = generate_token('dlm')
TOKEN_EN = generate_token('en')
TOKEN_NUM = generate_token('num')

TOKEN_NGRAM_PAD_BEGIN = generate_token('ngb')
TOKEN_NGRAM_PAD_END = generate_token('nge')

TOKEN_BMES_BREAK = 'BMES_BREAK'
