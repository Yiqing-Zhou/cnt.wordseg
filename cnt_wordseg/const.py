def generate_token(tag, add_slash=False):
    return f'<{"/" if add_slash else ""}{tag}>'


TOKEN_DLM = generate_token('dlm')
TOKEN_EN = generate_token('en')
TOKEN_NUM = generate_token('num')
