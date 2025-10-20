import numbers

def _elemwise(x, y, op):
    # ตัวเลข
    if isinstance(x, numbers.Number) and isinstance(y, numbers.Number):
        return op(x, y)
    # ลิสต์
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            raise ValueError(f"Shape mismatch in list: {len(x)} != {len(y)}")
        return [_elemwise(xi, yi, op) for xi, yi in zip(x, y)]
    # ทูเพิล
    if isinstance(x, tuple) and isinstance(y, tuple):
        if len(x) != len(y):
            raise ValueError(f"Shape mismatch in tuple: {len(x)} != {len(y)}")
        return tuple(_elemwise(xi, yi, op) for xi, yi in zip(x, y))
    # ชนิดไม่ตรงกัน
    raise TypeError(f"Type mismatch: {type(x).__name__} vs {type(y).__name__}")

def minus_pairs(p1, p2):
    a1, b1 = p1
    a2, b2 = p2
    return (
        _elemwise(a1, a2, lambda u, v: u - v),
        _elemwise(b1, b2, lambda u, v: u - v),
    )

def add_pairs(p1, p2):
    a1, b1 = p1
    a2, b2 = p2
    return (
        _elemwise(a1, a2, lambda u, v: u + v),
        _elemwise(b1, b2, lambda u, v: u + v),
    )

def zeros_like(x, zero=0):
    if isinstance(x, list):
        return [zeros_like(e, zero) for e in x]
    elif isinstance(x, tuple):
        return tuple(zeros_like(e, zero) for e in x)
    else:
        return zero
    
def scale(x, k):
    if isinstance(x, list):
        return [scale(e, k) for e in x]
    if isinstance(x, tuple):
        return tuple(scale(e, k) for e in x)
    if isinstance(x, numbers.Number):
        return x * k
    return x