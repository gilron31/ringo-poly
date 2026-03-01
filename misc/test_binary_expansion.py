import binary_expansion


def test_1():
    assert not binary_expansion.compute_binary_expansion(10, 5) is None
    assert binary_expansion.compute_binary_expansion(10, 4) is None
    assert not binary_expansion.compute_binary_expansion(123123123, 43) is None
    assert binary_expansion.compute_binary_expansion(123123123, 42) is None
