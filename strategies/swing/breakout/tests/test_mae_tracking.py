from strategies.swing.breakout.models import SetupInstance


def test_setup_instance_has_excursion_fields():
    setup = SetupInstance(symbol="SPY")  # use minimal constructor
    assert hasattr(setup, "mfe_price")
    assert hasattr(setup, "mae_price")
    assert hasattr(setup, "mfe_r")
    assert hasattr(setup, "mae_r")


def test_excursion_fields_default_to_zero():
    setup = SetupInstance(symbol="SPY")
    assert setup.mfe_price == 0.0
    assert setup.mae_price == 0.0
    assert setup.mfe_r == 0.0
    assert setup.mae_r == 0.0
