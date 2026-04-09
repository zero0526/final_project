def ema(prev: float, curr: float, n: int= 0):
    if n<0:
        raise ValueError('n must be non-negative')
    alpha= 1/(2+n)
    return prev*(1-alpha) + curr*alpha