from allennlp.common.testing import AllenNlpTestCase
from cnt.wordseg.data_reader import BMESContextTaggingDatasetReader

from os.path import dirname, join


FIXTURES_FODLER = join(dirname(__file__), 'fixtures')


class TestBMESContextTaggingDatasetReader(AllenNlpTestCase):

    def test_loading(self):
        path = join(FIXTURES_FODLER, 'example_data.txt')
        reader = BMESContextTaggingDatasetReader()
        instances = reader.read(path)

        assert len(instances) == 10
        instance0 = instances[0]
        assert len(instance0.fields['tokens'].tokens) == len(instance0.fields['tags'].labels)
        assert len(instance0.fields['context'].tokens) == 1
