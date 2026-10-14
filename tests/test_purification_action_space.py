"""
Checks for the joint routing + purification action space.

Run from the repository root (the quantum link fidelity table is loaded relative to
the working directory):

    PYTHONPATH=src python tests/test_purification_action_space.py
"""

import numpy as np

from env.entanglementenv import EntanglementEnv
from env.quantum_network import QuantumNetwork


def make_env(max_purification_rounds, purification_cost_weight=0.0, reward_type="fidelity"):
    network = QuantumNetwork(
        n_nodes=10,
        n_nodes_connected=10,
        random_topology=0,
        n_quantum_links=10,
        episode_steps=20,
        neighbor_count=3,
        ttl=20,
    )
    env = EntanglementEnv(
        network,
        n_data=2,
        ttl=20,
        enable_action_mask=True,
        max_purification_rounds=max_purification_rounds,
        purification_cost_weight=purification_cost_weight,
        reward_type=reward_type,
    )
    env.reset()
    return env


def test_action_space_size():
    """P = 0 must reproduce the original action space exactly."""
    env0 = make_env(0)
    assert env0.action_space.n == env0.network.neighbor_count + 1, env0.action_space.n
    assert env0.action_mask.shape == (2, env0.network.neighbor_count + 1)

    for p in (1, 2, 3, 5):
        env = make_env(p)
        expected = 1 + env.network.neighbor_count * (p + 1)
        assert env.action_space.n == expected, (p, env.action_space.n, expected)
        assert env.action_mask.shape == (2, expected)
    print("ok  action space size")


def test_encode_decode_roundtrip():
    """Every (edge, rounds) pair maps to a unique action and back."""
    for p in (0, 1, 3, 4):
        env = make_env(p)
        seen = set()
        assert env.decode_action(0) == (-1, 0)
        assert env.encode_action(-1) == 0
        for edge in range(env.network.neighbor_count):
            for rounds in range(p + 1):
                a = env.encode_action(edge, rounds)
                assert 1 <= a < env.action_space.n, (p, edge, rounds, a)
                assert a not in seen, f"collision at {a}"
                seen.add(a)
                assert env.decode_action(a) == (edge, rounds), (p, edge, rounds)
        # the non-idle actions exactly cover 1 .. n-1
        assert seen == set(range(1, env.action_space.n))
    print("ok  encode/decode roundtrip")


def test_p0_decode_matches_original():
    """With P = 0, decoding is the original `edge = action - 1` rule."""
    env = make_env(0)
    for a in range(1, env.action_space.n):
        edge, rounds = env.decode_action(a)
        assert edge == a - 1 and rounds == 0, (a, edge, rounds)
    print("ok  P=0 decode is backwards compatible")


def test_mask_blocks_unaffordable_rounds():
    """
    A level r is legal only if the edge still has r + 1 unreserved links.
    Mask convention in this codebase: True (1) means *forbidden*.
    """
    p = 3
    env = make_env(p)
    request = env.requests[0]
    node = env.network.get_nodes(env.env_id)[request.now]

    # force a known number of available links on the first outgoing edge
    edge = env.network.edges[node.edges[0]]
    edge.reserved_links.clear()
    keep = 2
    while len(edge.links) > keep:
        edge.links.pop()
    assert len(edge.links) == keep, len(edge.links)

    request.visited_nodes = {request.now}
    env._calculate_action_mask(request, 0)

    for rounds in range(p + 1):
        a = env.encode_action(0, rounds)
        forbidden = bool(env.action_mask[0, a])
        # rounds r needs r + 1 links; keep = 2 links affords r = 0 and r = 1
        assert forbidden == (rounds + 1 > keep), (rounds, forbidden, keep)
    print("ok  mask blocks purification levels the edge cannot afford")


def test_schedule_stays_aligned_with_path():
    """request.distillations[j] must describe hop path[j] -> path[j + 1]."""
    env = make_env(2)
    env.reset()
    # main.py populates these during setup; a bare env built here has them unset
    if env.network.shortest_available_paths_weights is None:
        env.network.shortest_available_paths_weights = {}
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.pre_step()
        actions = np.zeros(env.n_requests, dtype=np.int32)
        for i in range(env.n_requests):
            legal = np.flatnonzero(~env.action_mask[i])
            actions[i] = rng.choice(legal) if len(legal) else 0
        env.step(actions)
        for request in env.requests:
            assert len(request.distillations) <= max(len(request.path) - 1, 0), (
                len(request.distillations),
                len(request.path),
            )
    print("ok  purification schedule stays aligned with the path")


def test_auto_distillation_disabled_is_noop():
    env = make_env(2)
    env.network.auto_distillation_threshold = 0
    env.network.auto_distillation_links_consumed = 0
    for edge in env.network.edges:
        before = len(edge.links)
        env.network.perform_auto_distillation(edge, 0)
        assert len(edge.links) == before
    assert env.network.auto_distillation_links_consumed == 0
    print("ok  auto-distillation with threshold 0 consumes nothing")


def test_reliq_default_never_writes_schedule():
    """P = 0 (RELiQ): random routing must leave every request's schedule empty."""
    env = make_env(0)
    env.reset()
    if env.network.shortest_available_paths_weights is None:
        env.network.shortest_available_paths_weights = {}
    rng = np.random.default_rng(1)
    for _ in range(30):
        env.pre_step()
        actions = np.zeros(env.n_requests, dtype=np.int32)
        for i in range(env.n_requests):
            legal = np.flatnonzero(~env.action_mask[i])
            actions[i] = rng.choice(legal) if len(legal) else 0
        env.step(actions)
        for request in env.requests:
            assert request.distillations == [], request.distillations
    assert env.done_metrics.total_purification_resources == 0
    print("ok  RELiQ default path never writes a purification schedule")


def _random_rollout(env, steps, seed):
    env.reset()
    if env.network.shortest_available_paths_weights is None:
        env.network.shortest_available_paths_weights = {}
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(steps):
        env.pre_step()
        actions = np.zeros(env.n_requests, dtype=np.int32)
        for i in range(env.n_requests):
            legal = np.flatnonzero(~env.action_mask[i])
            actions[i] = rng.choice(legal) if len(legal) else 0
        _, _, reward, _, _ = env.step(actions)
        rewards.extend(reward.tolist())
    return np.array(rewards)


def test_ratio_reward_bounded():
    """Ratio reward never exceeds 1 (no reward for breaking a swap chain early)."""
    for p in (0, 3):
        env = make_env(p, reward_type="ratio")
        for request in env.requests:
            assert getattr(request, "min_hops", 0) >= 1
        r = _random_rollout(env, 40, seed=p)
        positive = r[r > 0]
        assert np.all(positive <= 1.0 + 1e-9), positive.max()
    print("ok  ratio reward is bounded by 1")


if __name__ == "__main__":
    test_action_space_size()
    test_encode_decode_roundtrip()
    test_p0_decode_matches_original()
    test_mask_blocks_unaffordable_rounds()
    test_schedule_stays_aligned_with_path()
    test_auto_distillation_disabled_is_noop()
    test_reliq_default_never_writes_schedule()
    test_ratio_reward_bounded()
    print("\nall checks passed")
