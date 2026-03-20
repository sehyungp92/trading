from pathlib import Path

import yaml


def test_compose_defines_three_stock_trader_services():
    compose_path = Path(__file__).resolve().parents[1] / "infra" / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = compose["services"]
    assert set(services) == {"strategy_iaric", "strategy_orb", "strategy_alcb"}

    iaric = services["strategy_iaric"]
    us_orb = services["strategy_orb"]
    alcb = services["strategy_alcb"]

    # No profiles — all services run unconditionally
    for svc in (iaric, us_orb, alcb):
        assert "profiles" not in svc

    # Commands
    assert iaric["command"] == ["python", "-m", "strategy_iaric"]
    assert us_orb["command"] == ["python", "-m", "strategy_orb"]
    assert alcb["command"] == ["python", "-m", "strategy_alcb"]

    # network_mode: host (no bridge networks)
    for svc in (iaric, us_orb, alcb):
        assert svc["network_mode"] == "host"
        assert "networks" not in svc

    assert "networks" not in compose

    # Deploy mode
    for svc in (iaric, us_orb, alcb):
        assert svc["environment"]["STOCK_TRADER_DEPLOY_MODE"] == "${STOCK_TRADER_DEPLOY_MODE:-both}"

    # Capital allocation defaults sum to 100%
    for svc in (iaric, us_orb, alcb):
        assert svc["environment"]["STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT"] == "${STOCK_TRADER_CAPITAL_ALLOCATION_IARIC_PCT:-33.333333}"
        assert svc["environment"]["STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT"] == "${STOCK_TRADER_CAPITAL_ALLOCATION_US_ORB_PCT:-33.333333}"
        assert svc["environment"]["STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT"] == "${STOCK_TRADER_CAPITAL_ALLOCATION_ALCB_PCT:-33.333334}"

    # Volume mounts — data + instrumentation for each strategy
    # Use exact volume strings to prevent regression (e.g. mounting only a subdir)
    iaric_vols = " ".join(iaric["volumes"])
    orb_vols = " ".join(us_orb["volumes"])
    alcb_vols = " ".join(alcb["volumes"])

    assert ":/app/data/strategy_iaric" in iaric_vols
    assert ":/app/data/strategy_orb" in orb_vols
    assert ":/app/data/strategy_orb/" not in orb_vols  # must be full dir, not subdir
    assert ":/app/data/strategy_alcb" in alcb_vols
    for svc in (iaric, us_orb, alcb):
        assert "/app/instrumentation/data" in " ".join(svc["volumes"])
