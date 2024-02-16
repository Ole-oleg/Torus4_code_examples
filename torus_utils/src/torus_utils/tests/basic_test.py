from src.basic import get_branch_points


def test_get_branch_points() -> None:
    K = 0.5 + 0.0001j
    eta_true_values = (
        0.8758 - 0.4830j,
        13.4242 - 0.0002j,
        0.8756 + 0.4829j,
        0.0745 + 0.0000j,
    )

    eta_values = get_branch_points(K)

    for n, eta in enumerate(eta_values):
        assert abs(eta - eta_true_values[n]) < 1e-4
