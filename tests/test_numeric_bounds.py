import pytest

from synapse.storage import SynapseStorage
from synapse.stores.constants import bounded_strength, bounded_unit


def test_bounded_unit_clamps_and_defaults():
    assert bounded_unit(-0.5) == 0.0
    assert bounded_unit(2.0) == 1.0
    assert bounded_unit(None, default=0.7) == 0.7
    assert bounded_unit("0.25") == 0.25


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bounded_unit_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        bounded_unit(bad)


def test_bounded_strength_clamps_to_five():
    assert bounded_strength(9.0) == 5.0
    assert bounded_strength(-1.0) == 0.0


def test_add_node_rejects_nan_salience():
    storage = SynapseStorage(":memory:")
    with pytest.raises(ValueError):
        storage.add_node(
            uri="synapse://decision/nan",
            l0_abstract="nan probe",
            category="decision",
            salience=float("nan"),
        )
    storage.close()


def test_add_episode_rejects_nan_confidence():
    storage = SynapseStorage(":memory:")
    with pytest.raises(ValueError):
        storage.add_episode(
            title="nan episode",
            situation="s",
            action_taken="a",
            outcome="o",
            lessons_learned="l",
            confidence=float("nan"),
        )
    storage.close()
