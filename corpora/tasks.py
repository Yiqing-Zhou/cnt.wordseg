from os.path import basename, dirname, join, exists
from os import makedirs
from shutil import rmtree
from glob import glob
from collections import defaultdict
from itertools import chain
from functools import partial

from invoke import task


DATA_FOLDER = join(dirname(__file__), 'data')

DOWNLOAD_FOLDER = join(DATA_FOLDER, 'download')
FLATTEN_FOLDER = join(DATA_FOLDER, 'flatten')
SPACE_FOLDER = join(DATA_FOLDER, 'space')
BMES_FOLDER = join(DATA_FOLDER, 'bmes')
FINAL_FOLDER = join(DATA_FOLDER, 'final')

DATASET_USAGES = ['train', 'dev', 'test']


def split_line(line):
    return line.strip().split()


def join_line(components):
    return ' '.join(components)


def space_space(lines):
    return list(map(
        lambda l: join_line(split_line(l)),
        lines,
    ))


def pos_space(sep, lines):
    return list(map(
        lambda l: join_line(map(
            lambda component: component.split(sep)[0],
            split_line(l),
        )),
        lines,
    ))


def conll_space(lines):
    groups = []
    cur = []
    for line in lines:
        components = split_line(line)
        if not components:
            assert cur
            groups.append(join_line(cur))
            cur = []
        else:
            cur.append(components[1])
    if not cur:
        groups.append(join_line(cur))

    return groups


# name -> fn(lines)
# fn returns space-seperated lines (one for a sentences).
DATASET_TO_SPACE = {
    'as': space_space,
    'cityu': space_space,
    'ctb': space_space,
    'msr': space_space,
    'pku': space_space,
    'sxu': space_space,

    'udc': conll_space,
    'wtb': conll_space,

    'cnc': partial(pos_space, '/'),
    'zx': partial(pos_space, '_'),
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
            for kw in DATASET_USAGES:
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


def read_lines(path):
    with open(path) as fin:
        return list(map(lambda l: l.rstrip('\n'), fin.readlines()))


def dump_lines(path, lines):
    print(f'Dump {path}')
    with open(path, 'w') as fout:
        fout.write('\n'.join(lines))


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


@task
def to_space(c):
    init_folder(SPACE_FOLDER)

    for name, process in DATASET_TO_SPACE.items():
        for usage in DATASET_USAGES:
            flatten_path = dataset_flatten_path(name, usage)
            if not exists(flatten_path):
                continue

            lines = read_lines(flatten_path)
            dump_lines(
                dataset_space_path(name, usage),
                process(lines)
            )
