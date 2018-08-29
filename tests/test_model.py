from allennlp.common.testing import ModelTestCase
from cnt.wordseg.model import CntWordSeg

from os.path import dirname, join


FIXTURES_FODLER = join(dirname(__file__), 'fixtures')


class TestCntWordSeg(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            join(FIXTURES_FODLER, 'cnt_wordseg.json'),
            join(FIXTURES_FODLER, 'example_data.txt')
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
