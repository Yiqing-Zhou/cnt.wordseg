from allennlp.data import Vocabulary
from allennlp.common.testing import AllenNlpTestCase
from cnt.wordseg.data_reader import WordSegTaggingDatasetReader

from os.path import dirname, join


FIXTURES_FODLER = join(dirname(__file__), 'fixtures')


class TestWordSegTaggingDatasetReader(AllenNlpTestCase):

    def test_loading(self):
        path = join(FIXTURES_FODLER, 'example_data.txt')

        # by default, generate context.
        reader = WordSegTaggingDatasetReader()
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

        # inject tags.
        reader = WordSegTaggingDatasetReader(inject_context=True)
        instances = reader.read(path)

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
