from os.path import basename, dirname, join, exists
from os import makedirs
from shutil import rmtree
from glob import glob
from collections import defaultdict
from itertools import chain

from invoke import task


DATA_FOLDER = join(dirname(__file__), 'data')

DOWNLOAD_FOLDER = join(DATA_FOLDER, 'download')
FLATTEN_FOLDER = join(DATA_FOLDER, 'flatten')
SPACE_FOLDER = join(DATA_FOLDER, 'space')
BMES_FOLDER = join(DATA_FOLDER, 'bmes')
FINAL_FOLDER = join(DATA_FOLDER, 'final')

DATASET_KEYWORDS = ['train', 'dev', 'test']
DATASET_CONFIG = {
    'as': 'space',
    'cityu': 'space',
    'cnc': 'pos /',
    'ctb': 'space',
    'msr': 'space',
    'pku': 'space',
    'sxu': 'space',
    'udc': 'conll',
    'wtb': 'conll',
    'zx': 'pos _',
}


def _dataset_filename(name, usage):
    return f'{name}_{usage}.txt'


def dataset_flatten_path(name, usage):
    return join(FLATTEN_FOLDER, _dataset_filename(name, usage))


def dataset_space_path(name, usage):
    return join(SPACE_FOLDER, _dataset_filename(name, usage))


def dataset_bmes_path(name, usage):
    return join(BMES_FOLDER, _dataset_filename(name, usage))


def dataset_final_path(name, usage):
    return join(FINAL_FOLDER, _dataset_filename(name, usage))


def init_folder(path):
    if exists(path):
        print(f'INIT: removing {path}')
        rmtree(path)
    print(f'INIT: creating {path}')
    makedirs(path)


def collect_dataset(folder, name_fn):

    def get_path_usage(folder):
        path_usage = []
        for path in glob(join(folder, '**'), recursive=True):
            name = basename(path)
            usage = None
            for kw in DATASET_KEYWORDS:
                if kw in name:
                    usage = kw
                    break
            if usage:
                path_usage.append(
                    (path, usage)
                )
        return path_usage

    dataset = defaultdict(list)
    for path, usage in get_path_usage(folder):
        name = name_fn(path, usage).lower()
        dataset[name].append(
            (path, usage),
        )
    return dataset


def flatten_sighan2005(root):
    return collect_dataset(
        join(root, 'sighan2005'),
        lambda path, usage: basename(path).split(usage)[0][:-1],
    )


def flatten_other(root):
    return collect_dataset(
        join(root, 'other'),
        lambda path, _: basename(dirname(path)),
    )


@task
def download(c):
    init_folder(DOWNLOAD_FOLDER)
    init_folder(FLATTEN_FOLDER)

    # download files from github.
    c.run(
        f'wget -O {join(DOWNLOAD_FOLDER, "sighan2005.tar.gz")}'
        f' https://github.com/cnt-dev/cnt.wordseg/releases/download/corpora/sighan2005.tar.gz'
    )
    c.run(
        f'wget -O {join(DOWNLOAD_FOLDER, "other.tar.gz")}'
        f' https://github.com/cnt-dev/cnt.wordseg/releases/download/corpora/other.tar.gz'
    )

    # uncompress and flatten.
    c.run(
        f'find {DOWNLOAD_FOLDER} -name "*.tar.gz"'
        f' -exec tar -xvf {{}} -C {FLATTEN_FOLDER} \;'
    )
    for name, path_usage in chain(
        flatten_sighan2005(FLATTEN_FOLDER).items(),
        flatten_other(FLATTEN_FOLDER).items(),
    ):
        for path, usage in path_usage:
            c.run(
                f'cp {path} {dataset_flatten_path(name, usage)}'
            )
