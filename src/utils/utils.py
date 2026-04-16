import re
import numpy as np

order_pattern = re.compile(r'(\d+)')

def convert_nodeid2order(nodeid: str)->int:
    match = order_pattern.search(nodeid)
    if match:
        return int(match.group(1))
    return -1

def one_hot(idx, dim):
    v = np.zeros(dim)
    v[idx] = 1
    return v


def to_binary(number: int, num_logit: int) -> np.ndarray:
    shifts = np.arange(num_logit - 1, -1, -1)

    binary_array = (number >> shifts) & 1

    return binary_array.astype(np.int8)


def from_binary(binary_array: np.ndarray) -> int:
    num_logit = len(binary_array)

    powers = 2 ** np.arange(num_logit - 1, -1, -1)

    number = np.dot(binary_array, powers)

    return int(number)

if __name__=='__main__':
    print(convert_nodeid2order('N123'))
