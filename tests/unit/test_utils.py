import os

from isco.utils.seeds import core_order_env, cores_for, idle_core_order


def test_idle_core_order_is_a_permutation_of_allowed_cores():
    order = idle_core_order(0.05)
    assert sorted(order) == sorted(os.sched_getaffinity(0))


def test_cores_for_takes_blocks_from_the_given_order():
    order = [5, 3, 1, 0, 2, 4]
    assert cores_for(1, 2, order) == [3, 5]
    assert cores_for(2, 2, order) == [0, 1]
    assert cores_for(4, 2, order) == [3, 5]          # wraps around


def test_core_order_env_is_scoped():
    before = os.environ.get("ISCO_CORE_ORDER")
    with core_order_env():
        assert sorted(int(c) for c in os.environ["ISCO_CORE_ORDER"].split(",")) == sorted(os.sched_getaffinity(0))
    assert os.environ.get("ISCO_CORE_ORDER") == before
