from typing import Dict, List
import logging
import json

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field  # noqa
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from .const import (
    TOKEN_NGRAM_PAD_BEGIN, TOKEN_NGRAM_PAD_END
)


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "/"
DEFAULT_TOKEN_DELIMITER = ' '


# from https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/sequence_tagging.py  # noqa
@DatasetReader.register("wordseg_tagging")
class WordSegTaggingDatasetReader(DatasetReader):

    def __init__(
        self,
        word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
        token_delimiter: str = DEFAULT_TOKEN_DELIMITER,
        disable_context: bool = False,
        inject_context_to_tokens: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None,
        context_indexers: Dict[str, TokenIndexer] = None,
        enable_bigram: bool = False,
        token_bigram_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False
    ) -> None:

        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}  # noqa
        self._context_indexers = context_indexers or {'contexts': SingleIdTokenIndexer(namespace='contexts')}  # noqa

        self._disable_context = disable_context
        self._inject_context_to_tokens = inject_context_to_tokens

        # TODO: consult to AllenNLP team.
        # `count_vocab_items` is conflict with n-gram token indexer.
        # Use `SingleIdTokenIndexer` to represent bigram for now.
        self._enable_bigram = enable_bigram
        if self._enable_bigram:
            self._token_bigram_indexers = token_bigram_indexers or {'tokens_bigram': SingleIdTokenIndexer(namespace='tokens_bigram')}  # noqa

        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:

            logger.info("Reading instances from lines in file at: %s", file_path)  # noqa
            for line in data_file:
                line = line.strip("\n")

                # skip blank lines
                if not line:
                    continue

                line = json.loads(line)
                context = line['context']
                bmes_seq = line['bmes_seq']

                tokens_and_tags = [
                    pair.rsplit(self._word_tag_delimiter, 1)
                    for pair in bmes_seq.split(self._token_delimiter)
                ]
                tokens = [Token(token) for token, _ in tokens_and_tags]
                tags = [tag for _, tag in tokens_and_tags]

                data_dict = {
                    'tokens': tokens,
                    'tags': tags,
                }

                if not self._inject_context_to_tokens:
                    if not self._disable_context:
                        data_dict['context'] = Token(context)
                else:
                    start_tok = context
                    end_tok = list(context)
                    end_tok.insert(1, '/')
                    end_tok = ''.join(end_tok)

                    tokens.insert(0, Token(start_tok))
                    tokens.append(Token(end_tok))

                    tags.insert(0, 'S-IGN')
                    tags.append('S-IGN')

                    # update tokens & tags.
                    data_dict['tokens'] = tokens
                    data_dict['tags'] = tags

                if self._enable_bigram:
                    tokens_bigram = []

                    seqlen = len(tokens)
                    for idx in range(seqlen + 1):
                        if idx == 0:
                            pre_text = TOKEN_NGRAM_PAD_BEGIN
                            cur_text = tokens[idx].text
                        elif idx == seqlen:
                            pre_text = tokens[-1].text
                            cur_text = TOKEN_NGRAM_PAD_END
                        else:
                            pre_text = tokens[idx - 1].text
                            cur_text = tokens[idx].text
                        # generate bigram token.
                        tokens_bigram.append(Token(f'{pre_text}{cur_text}'))
                    # add to dict.
                    data_dict['tokens_bigram_fwd'] = tokens_bigram[:-1]
                    data_dict['tokens_bigram_bwd'] = tokens_bigram[1:]

                yield self.text_to_instance(**data_dict)

    def text_to_instance(
        self,
        tokens: List[Token],
        tokens_bigram_fwd: List[Token] = None, tokens_bigram_bwd: List[Token] = None,
        tags: List[str] = None,
        context: Token = None
    ) -> Instance:  # noqa type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """  # noqa
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        if self._inject_context_to_tokens or self._disable_context:
            assert context is None
        else:
            assert context
            fields['context'] = TextField([context], self._context_indexers)

        if self._enable_bigram:
            assert tokens_bigram_fwd and tokens_bigram_bwd
            fields['tokens_bigram_fwd'] = TextField(tokens_bigram_fwd, self._token_bigram_indexers)
            fields['tokens_bigram_bwd'] = TextField(tokens_bigram_bwd, self._token_bigram_indexers)

        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)

        return Instance(fields)
