from os.path import basename, dirname, join, exists
from os import makedirs
from shutil import rmtree
from glob import glob
from collections import defaultdict
from itertools import chain
from functools import partial
import random
import json

from invoke import task
from tqdm import tqdm

from cnt.wordseg.utils import (
    break_to_sentences,
    break_to_segments,
    preprocess_segments,
    extract_chars,
    split_bmes_lines,
)
from cnt.wordseg.const import (
    generate_token,
    TOKEN_BMES_BREAK,
)


DATA_FOLDER = join(dirname(__file__), 'data')

# 1. for download and flattening.
DOWNLOAD_FOLDER = join(DATA_FOLDER, 'download')
FLATTEN_FOLDER = join(DATA_FOLDER, 'flatten')
# 2. for converting all files to space-seperated format.
SPACE_FOLDER = join(DATA_FOLDER, 'space')
# 3. for processing.
PROCESSED_FOLDER = join(DATA_FOLDER, 'processed')
# 4. to BMES format.
BMES_FOLDER = join(DATA_FOLDER, 'bmes')
BMES_ALLENNLP_FOLDER = join(DATA_FOLDER, 'bmes_allennlp')
# 5. final usage.
FINAL_FOLDER = join(DATA_FOLDER, 'final')
FINAL_ALLENNLP_FOLDER = join(DATA_FOLDER, 'final_allennlp')

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


def create_path_generator(folder):
    return lambda name, usage: join(folder, f'{name}_{usage}.txt')


dataset_flatten_path = create_path_generator(FLATTEN_FOLDER)
dataset_space_path = create_path_generator(SPACE_FOLDER)
dataset_processed_path = create_path_generator(PROCESSED_FOLDER)
dataset_bmes_path = create_path_generator(BMES_FOLDER)
dataset_final_path = create_path_generator(FINAL_FOLDER)
dataset_bmes_allennlp_path = create_path_generator(BMES_ALLENNLP_FOLDER)
dataset_final_allennlp_path = create_path_generator(FINAL_ALLENNLP_FOLDER)


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
    print(f'Load {path}')
    with open(path) as fin:
        return list(map(lambda l: l.rstrip('\n'), fin.readlines()))


def dump_lines(path, lines):
    print(f'Dump {path}')
    lines = filter(bool, lines)
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
    for line in tqdm(lines):
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
        print(f'Processing dataset {name}')

        print('train')
        train = apply_process_fn(dataset[name]['train'], process_fn)

        print('dev')
        dev = apply_process_fn(dataset[name]['dev'], process_fn)

        print('test')
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


def _bmes_line(c, tag, tag_dlm='\t'):
    return f'{c}{tag_dlm}{tag}'


def _bmes_out(c, tag, out, tag_dlm):
    out.append(_bmes_line(c, tag, tag_dlm))


def _word2bmes(word, out, tag_dlm):
    chars = extract_chars(word)

    if len(chars) == 1:
        _bmes_out(word, 'S', out, tag_dlm)
        return

    else:
        _bmes_out(chars[0], 'B', out, tag_dlm)
        for c in chars[1:-1]:
            _bmes_out(c, 'M', out, tag_dlm)
        _bmes_out(chars[-1], 'E', out, tag_dlm)


def _bmes(folder_out, path_fn, tag_dlm='\t', c_dlm=None):
    init_folder(folder_out)

    for name in DATASET_KEYS:
        for usage in DATASET_USAGES:
            path = dataset_processed_path(name, usage)
            assert exists(path)

            out = []
            for line in read_lines(path):
                line_out = []
                for word in split_line(line):
                    _word2bmes(word, line_out, tag_dlm)

                if c_dlm:
                    line_out = [c_dlm.join(line_out)]
                else:
                    line_out.append(TOKEN_BMES_BREAK)

                out.extend(line_out)

            dump_lines(
                path_fn(name, usage),
                out,
            )


@task
def bmes(c):
    # [word, ...]
    # ->
    # c\t(B|M|E|S)\n ...
    _bmes(
        BMES_FOLDER, dataset_bmes_path,
    )


@task
def bmes_allennlp(c):
    # [word, ...]
    # ->
    # word/(B|M|E|S) ... \n
    _bmes(
        BMES_ALLENNLP_FOLDER, dataset_bmes_allennlp_path,
        tag_dlm='/', c_dlm=' ',
    )


def _merge_files(path_fn, merged_name, usage):
    merged_path = path_fn(merged_name, usage)
    print(f'Merging to {merged_path}')

    with open(merged_path, 'w') as fout:
        for name in DATASET_KEYS:
            with open(path_fn(name, usage)) as fin:
                for line in fin:
                    fout.write(line)


@task
def final(c, merged_name='all'):
    init_folder(FINAL_FOLDER)

    def join_groups(groups):
        for group in groups:
            for line in group:
                yield line
            yield TOKEN_BMES_BREAK

    for usage in DATASET_USAGES:
        # add context tag to each dataset.
        for name in DATASET_KEYS:
            path = dataset_bmes_path(name, usage)
            assert exists(path)

            groups = split_bmes_lines(
                read_lines(path),
                _bmes_line(generate_token(name), 'S'),
                _bmes_line(generate_token(name, add_slash=True), 'S'),
            )

            dump_lines(
                dataset_final_path(name, usage),
                join_groups(groups),
            )

        _merge_files(dataset_final_path, merged_name, usage)


@task
def final_allennlp(c, merged_name='all'):
    # line
    # ->
    # {"context": "{name}", "bmes_seq": "bmes_seq"}
    init_folder(FINAL_ALLENNLP_FOLDER)

    for usage in DATASET_USAGES:
        for name in DATASET_KEYS:
            path = dataset_bmes_allennlp_path(name, usage)
            assert exists(path)

            out = []
            for line in read_lines(path):
                out.append(json.dumps(
                    {
                        "context": generate_token(name),
                        "bmes_seq": line,
                    },
                    ensure_ascii=False,
                ))

            dump_lines(
                dataset_final_allennlp_path(name, usage),
                out,
            )

        _merge_files(dataset_final_allennlp_path, merged_name, usage)


@task
def clean(c, include_final=False):
    for folder in [
        DOWNLOAD_FOLDER,
        FLATTEN_FOLDER,
        SPACE_FOLDER,
        PROCESSED_FOLDER,
        BMES_FOLDER,
    ]:
        init_folder(folder)

    if include_final:
        init_folder(FINAL_FOLDER)
