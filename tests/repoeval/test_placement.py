import pytest

from missy.repoeval.placement import (
    CapacityBudget,
    PlacementError,
    PoolCapacity,
    ResourceEnvelope,
    capacity_budget,
    select_pool,
)


def test_staging_selection_is_read_only_and_preserves_envelope():
    pools = [PoolCapacity("wide", 4000, 8000, 8000), PoolCapacity("staging", 2000, 2048, 1024)]
    before = list(pools)
    assert select_pool(ResourceEnvelope(1000, 1024, 512), pools).name == "staging"
    assert pools == before


def test_capacity_budget_considers_all_eligible_pools_without_using_them_as_targets():
    pools = [
        PoolCapacity("staging", 1200, 2048, 1024),
        PoolCapacity("available", 4000, 8192, 4096),
        PoolCapacity("disabled", 100000, 100000, 100000, False),
    ]
    assert capacity_budget(pools) == CapacityBudget(5200, 10240, 5120)
    assert select_pool(ResourceEnvelope(1000, 1024, 512), pools).name == "staging"


@pytest.mark.parametrize("stage", [(999, 2048, 1024), (2000, 1023, 1024), (2000, 2048, 511)])
def test_never_spills_when_staging_is_insufficient(stage):
    pools = [PoolCapacity("staging", *stage), PoolCapacity("other", 10000, 10000, 10000)]
    with pytest.raises(PlacementError, match="staging"):
        select_pool(ResourceEnvelope(1000, 1024, 512), pools)


def test_refuses_ineligible_or_insufficient_pools():
    with pytest.raises(PlacementError, match="staging"):
        select_pool(
            ResourceEnvelope(1000, 1024, 512),
            [
                PoolCapacity("staging", 2000, 2048, 1024, False),
                PoolCapacity("other", 2000, 2048, 1024),
            ],
        )
    with pytest.raises(PlacementError, match="staging"):
        select_pool(ResourceEnvelope(1000, 1024, 512), [PoolCapacity("other", 2000, 2048, 1024)])
    with pytest.raises(PlacementError, match="staging"):
        select_pool(ResourceEnvelope(1000, 1024, 512), [])
    with pytest.raises(PlacementError, match="duplicate"):
        capacity_budget([PoolCapacity("staging", 2000, 2048, 1024)] * 2)


def test_rejects_invalid_capacity_and_envelope():
    with pytest.raises(PlacementError):
        PoolCapacity("x", -1, 2000, 1000)
    with pytest.raises(PlacementError):
        ResourceEnvelope(0, 1000, 1000)
    with pytest.raises(PlacementError):
        ResourceEnvelope(True, 1000, 1000)
