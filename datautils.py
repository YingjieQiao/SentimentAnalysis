from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def get_train_data(
    filename: str,
    add_start_stop_states: bool = True,
) -> Tuple[List[List[str]], List[List[int]]]:
    filename = Path(filename)
    assert filename.is_file()

    with filename.open() as f:
        lines = f.readlines()
        f.close()

    lines = [line.strip().split() for line in lines]

    x: List[List[str]] = []  # list of list of tokens
    y: List[List[str]] = []  # list of list of labels

    tmp_x, tmp_y = [], []  # temporary list of tokens and labels

    for line in lines:
        if len(line) == 2:
            tmp_x.append(line[0])
            tmp_y.append(line[1])
        elif len(line) == 3:
            # Special token: ". ..."
            tmp_x.append(f"{line[0]} {line[1]}")
            tmp_y.append(line[2])
        elif len(line) == 0:
            x.append(tmp_x)

            if add_start_stop_states:
                tmp_y.insert(0, "<START>")
                tmp_y.append("<END>")

            y.append(tmp_y)
            tmp_x, tmp_y = [], []
        else:
            raise ValueError("Unexpected line format")

    return x, y


def get_test_data(filename: str) -> List[List[str]]:
    filename = Path(filename)
    assert filename.is_file()

    with filename.open() as f:
        lines = f.readlines()
        f.close()

    lines = [line.strip() for line in lines]
    x: List[List[str]] = []  # list of list of tokens
    tmp_x = []  # temporary list of tokens

    for line in lines:
        if len(line) == 0 and len(tmp_x) > 0:
            x.append(tmp_x)
            tmp_x = []
        else:
            tmp_x.append(line)

    if len(tmp_x) > 0:
        x.append(tmp_x)

    return x


def build_vocab(data: List[List[str]], use_unk: bool = False) -> Dict[str, int]:
    """
    Build vocabulary from training data.

    Args:
        data: list of list of tokens

    return:
        dict of token to index
    """
    counter = Counter()
    for i in data:
        counter.update(i)
    vocab = [t for t, _ in counter.most_common()]
    if use_unk:
        vocab.insert(0, "<UNK>")
    vocab = {t: i for i, t in enumerate(vocab)}
    return vocab


def encode_data(
    data: List[List[str]],
    vocab: Dict[str, int],
    unk_idx: int = 0,
) -> List[List[int]]:
    encoded_data = []
    for i in data:
        encoded_data.append([vocab.get(t, unk_idx) for t in i])
    return encoded_data
