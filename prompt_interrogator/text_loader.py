import os
from typing import List


def _load_file(filename: str):
    if not os.path.isfile(filename):
        raise ValueError('No such word list file')
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        return [line.strip() for line in f.readlines()]


def load_words(list_name: str) -> List[str]:
    base_dir = os.path.dirname(__file__)
    words = _load_file(f'{base_dir}/text/{list_name}.txt')
    if list_name == 'artists':
        words = ['by ' + word for word in words]
    return words
