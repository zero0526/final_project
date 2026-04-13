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
if __name__=='__main__':
    print(convert_nodeid2order('N123'))
