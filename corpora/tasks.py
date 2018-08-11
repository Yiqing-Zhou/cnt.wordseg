from os.path import basename, dirname, join, exists
from os import makedirs
from shutil import rmtree
from glob import glob
from collections import defaultdict
from itertools import chain
from functools import partial
import random

from invoke import task

from cnt_wordseg.utils import (
    break_to_sentences,
    break_to_segments,
    preprocess_segments,
)


DATA_FOLDER = join(dirname(__file__), 'data')

# for download and flattening.
DOWNLOAD_FOLDER = join(DATA_FOLDER, 'download')
FLATTEN_FOLDER = join(DATA_FOLDER, 'flatten')
# for converting all files to space-seperated format.
SPACE_FOLDER = join(DATA_FOLDER, 'space')
# for processing.
PROCESSED_FOLDER = join(DATA_FOLDER, 'processed')
# for final usage.
BMES_FOLDER = join(DATA_FOLDER, 'bmes')

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
DATASET_TO_SPACE_FNS = {
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
DATASET_KEYS = list(DATASET_TO_SPACE_FNS.keys())


def _dataset_filename(name, usage):
    return f'{name}_{usage}.txt'


def dataset_flatten_path(name, usage):
    return join(FLATTEN_FOLDER, _dataset_filename(name, usage))


def dataset_space_path(name, usage):
    return join(SPACE_FOLDER, _dataset_filename(name, usage))


def dataset_bmes_path(name, usage):
    return join(BMES_FOLDER, _dataset_filename(name, usage))


def dataset_processed_path(name, usage):
    return join(PROCESSED_FOLDER, _dataset_filename(name, usage))


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
        return list(filter(
            bool,
            map(lambda l: l.rstrip('\n'), fin.readlines()),
        ))


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
        f' https://github.com/cnt-dev/cnt.wordseg/releases/download/corpora/sighan2005.tar.gz'  # noqa: E501
    )
    c.run(
        f'wget -O {join(DOWNLOAD_FOLDER, "other.tar.gz")}'
        f' https://github.com/cnt-dev/cnt.wordseg/releases/download/corpora/other.tar.gz'  # noqa: E501
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

    for name, process in DATASET_TO_SPACE_FNS.items():
        for usage in DATASET_USAGES:
            flatten_path = dataset_flatten_path(name, usage)
            if not exists(flatten_path):
                continue

            lines = read_lines(flatten_path)
            dump_lines(
                dataset_space_path(name, usage),
                process(lines)
            )


# default mode:
#
# 1. use sentseg (enable_comma) to break down a sentence.
# 2. use dlmseg to extract segments.
# 3. deal with non-chinese chars.
#   3.1. replace all english chars with TOKEN_EN.
#   3.2. replace all digits with TOKEN_NUM.
#   3.3. insert TOKEN_DLM between discontinuous segments.
#   3.4. merge continuous special tokens.
# 4. remove any segments that doesn't contain chinese chars.
def default_process(line):
    ret = []
    for sent in break_to_sentences(line):
        segs = preprocess_segments(
            break_to_segments(sent),
            gap=1,
        )
        if not segs:
            continue
        ret.append(join_line(segs))
    return ret


def apply_process_fn(lines, process_fn):
    ret = []
    for line in lines:
        ret.extend(process_fn(line))
    return ret


@task
def process(
    c,
    # random seed (int) for reproduction.
    random_seed=None,
    # merge & randomize & regenerate train/dev/test.
    merge=True,
    # dev/test ratio.
    # if merge=False, test_ratio will be ignored and dev_ratio
    # will be applied to train set if dev set doesn't exist.
    dev_ratio=0.1, test_ratio=0.1,
    # preprocess mode.
    mode='default'
):
    # set random seed.
    if random_seed:
        random_seed = int(random_seed)
    else:
        random_seed = random.randint(0, 2**32 - 1)
    print(f'random seed: {random_seed}')
    random.seed(random_seed)

    # clean up.
    init_folder(PROCESSED_FOLDER)

    dataset = {}
    for name in DATASET_KEYS:
        dataset[name] = {}
        for usage in DATASET_USAGES:
            path = dataset_space_path(name, usage)
            if not exists(path):
                dataset[name][usage] = []
            else:
                dataset[name][usage] = read_lines(path)

    process_fn = None
    if mode == 'default':
        process_fn = default_process

    # cases:
    # 1. all train/dev/test exists.
    # 2. missing dev.
    for name in DATASET_KEYS:
        train = apply_process_fn(dataset[name]['train'], process_fn)
        dev = apply_process_fn(dataset[name]['dev'], process_fn)
        test = apply_process_fn(dataset[name]['test'], process_fn)

        # merge mode.
        if merge:
            full = train + dev + test
            random.shuffle(full)

            dev_size = int(len(full) * dev_ratio)
            test_size = int(len(full) * test_ratio)

            dev, full = full[:dev_size], full[dev_size:]
            test, full = full[:test_size], full[test_size:]
            train = full

        else:
            assert train and test
            if not dev:
                random.shuffle(train)
                dev_size = int(len(train) * dev_ratio)
                dev, train = train[:dev_size], train[dev_size:]

        # dump.
        dump_lines(
            dataset_processed_path(name, 'train'),
            train,
        )
        dump_lines(
            dataset_processed_path(name, 'dev'),
            dev,
        )
        dump_lines(
            dataset_processed_path(name, 'test'),
            test,
        )
