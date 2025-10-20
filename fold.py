def make_kfold(lst, k):
    n = len(lst)
    k = min(k, n)

    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        part = lst[start:start+fs]
        folds.append(part[0] if len(part) == 1 else part)
        start += fs

    return folds

def flatten_one_level(lst):
    out = []
    for sub in lst:
        if isinstance(sub, list):
            out.extend(sub)
        else:
            out.append(sub)
    return out