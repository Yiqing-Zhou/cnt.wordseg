from allennlp.data import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from cnt.wordseg.data_reader import WordSegTaggingDatasetReader

from os.path import dirname, join


FIXTURES_FODLER = join(dirname(__file__), 'fixtures')
EXAMPLE_DATA_PATH = join(FIXTURES_FODLER, 'example_data.txt')


class TestWordSegTaggingDatasetReader(AllenNlpTestCase):

    def test_default(self):
        # by default, generate context.
        reader = WordSegTaggingDatasetReader()
        instances = reader.read(EXAMPLE_DATA_PATH)

        assert len(instances) == 10
        instance0 = instances[0]
        assert len(instance0.fields['tokens'].tokens) == len(instance0.fields['tags'].labels)
        assert len(instance0.fields['context'].tokens) == 1

        # vocab test.
        vocab = Vocabulary.from_instances(instances)
        assert vocab.get_index_to_token_vocabulary("tokens")
        assert vocab.get_index_to_token_vocabulary("contexts")
        assert vocab.get_index_to_token_vocabulary("labels")

    def test_inject_context_to_tokens(self):
        # inject tags.
        reader = WordSegTaggingDatasetReader(inject_context_to_tokens=True)
        instances = reader.read(EXAMPLE_DATA_PATH)

        # make sure context not exists.
        vocab = Vocabulary.from_instances(instances)
        assert 'contexts' not in vocab._token_to_index

        # check tags.
        instance0 = instances[0]

        assert '<as>' == instance0.fields['tokens'].tokens[0].text
        assert '</as>' == instance0.fields['tokens'].tokens[-1].text
        assert 2 < len(instance0.fields['tokens'].tokens)

        assert 'S-IGN' == instance0.fields['tags'].labels[0]
        assert 'S-IGN' == instance0.fields['tags'].labels[-1]

    def test_inject_context_to_tokens_with_bigram(self):
        # inject tags & bigram
        reader = WordSegTaggingDatasetReader(inject_context_to_tokens=False, enable_bigram=True)
        instances = reader.read(EXAMPLE_DATA_PATH)

        # make sure tokens_bigram exists.
        vocab = Vocabulary.from_instances(instances)
        assert 'tokens_bigram' in vocab._token_to_index

        instance0 = instances[0]
        assert [
            '<ngb>他', '他们', '们唱'
        ] == [t.text for t in instance0.fields['tokens_bigram_fwd'].tokens]
        assert [
            '他们', '们唱', '唱<nge>'
        ] == [t.text for t in instance0.fields['tokens_bigram_bwd'].tokens]
