from allennlp.data import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from cnt.wordseg.data_reader import WordsegTaggingDatasetReader

from os.path import dirname, join


FIXTURES_FODLER = join(dirname(__file__), 'fixtures')


class TestWordsegTaggingDatasetReader(AllenNlpTestCase):

    def test_loading(self):
        path = join(FIXTURES_FODLER, 'example_data.txt')
        reader = WordsegTaggingDatasetReader()
        instances = reader.read(path)

        assert len(instances) == 10
        instance0 = instances[0]
        assert len(instance0.fields['tokens'].tokens) == len(instance0.fields['tags'].labels)
        assert len(instance0.fields['context'].tokens) == 1

        # vocab test.
        vocab = Vocabulary.from_instances(instances)
        assert vocab.get_index_to_token_vocabulary("tokens")
        assert vocab.get_index_to_token_vocabulary("contexts")
        assert vocab.get_index_to_token_vocabulary("labels")
