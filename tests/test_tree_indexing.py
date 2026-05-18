"""Sanity tests for the BFS-array tree indexing scheme."""


def parent(i: int) -> int:
    return (i - 1) // 2


def left(i: int) -> int:
    return 2 * i + 1


def right(i: int) -> int:
    return 2 * i + 2


def total_nodes(depth: int) -> int:
    return (1 << (depth + 1)) - 1


def leaf_offset(depth: int) -> int:
    return (1 << depth) - 1


def test_total_nodes_small():
    assert total_nodes(0) == 1
    assert total_nodes(1) == 3
    assert total_nodes(4) == 31
    assert total_nodes(20) == (1 << 21) - 1


def test_leaf_offset():
    assert leaf_offset(0) == 0
    assert leaf_offset(4) == 15
    assert leaf_offset(20) == (1 << 20) - 1


def test_parent_child_inverses():
    for i in range(1, 100):
        assert parent(left(i)) == i
        assert parent(right(i)) == i
