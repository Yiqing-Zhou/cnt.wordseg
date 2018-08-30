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
        token_indexers: Dict[str, TokenIndexer] = None,
        context_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False
    ) -> None:

        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}  # noqa
        self._context_indexers = context_indexers or {'contexts': SingleIdTokenIndexer(namespace='contexts')}  # noqa
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

                try:
                    line = json.loads(line)
                except:
                    print(line)
                    raise

                context = line['context']
                bmes_seq = line['bmes_seq']

                context = Token(context)

                tokens_and_tags = [
                    pair.rsplit(self._word_tag_delimiter, 1)
                    for pair in bmes_seq.split(self._token_delimiter)
                ]
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]

                yield self.text_to_instance(context, tokens, tags)

    def text_to_instance(self, context: Token, tokens: List[Token], tags: List[str] = None) -> Instance:  # noqa type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """  # noqa
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        fields['context'] = TextField([context], self._context_indexers)

        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)

        return Instance(fields)
