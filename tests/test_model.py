from allennlp.common.testing import ModelTestCase
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


class TestCntWordSegInjectTag(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            join(FIXTURES_FODLER, 'cnt_wordseg_inject_tag.json'),
            join(FIXTURES_FODLER, 'example_data.txt')
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


class TestCntWordSegBenchmarkGoogle(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            join(FIXTURES_FODLER, 'cnt_wordseg_benchmark_google.json'),
            join(FIXTURES_FODLER, 'example_data.txt')
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


class TestCntWordSegBenchmarkGoogleWithContext(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            join(FIXTURES_FODLER, 'cnt_wordseg_benchmark_google_with_context.json'),
            join(FIXTURES_FODLER, 'example_data.txt')
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
