def test_exp_run_keeps_preregistered_defaults(monkeypatch):
    """`isco exp run` passes budget/seeds only when given, so every experiment runs with
    its own pre-registered defaults (E1 budget 40, E2 60, E8 seeds 0,1, ...)."""
    from typer.testing import CliRunner

    from isco.cli.main import app
    from isco.eval import hypotheses as H

    seen = {}

    def fake(parts, n_workers=20, out=None, budget=40, seeds=(0,), **kw):
        seen.update(parts=parts, budget=budget, seeds=list(seeds))
        return "x"

    monkeypatch.setattr(H, "e_fake", fake, raising=False)
    r = CliRunner().invoke(app, ["exp", "run", "e_fake", "--parts", "tless_13"])
    assert r.exit_code == 0, r.output
    assert seen == {"parts": ["tless_13"], "budget": 40, "seeds": [0]}
    r = CliRunner().invoke(app, ["exp", "run", "e_fake", "--parts", "tless_13", "--budget", "12", "--seeds", "0,1"])
    assert r.exit_code == 0 and seen["budget"] == 12 and seen["seeds"] == [0, 1]
