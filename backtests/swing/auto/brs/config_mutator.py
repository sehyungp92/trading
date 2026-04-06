"""BRS config mutation — dot-notation for BRSConfig.

Follows swing auto/config_mutator.py pattern:
  - "symbol_configs.QQQ.adx_on": 18  → mutate per-symbol config
  - "param_overrides.disable_s1": 1  → merge into param_overrides
  - "slippage.commission_per_share_etf": 0.005  → mutate slippage
  - "adx_strong": 30  → direct top-level replace
"""
from __future__ import annotations

from dataclasses import fields, replace

from research.backtests.swing.config_brs import BRSConfig, BRSSymbolConfig


def mutate_brs_config(base: BRSConfig, mutations: dict) -> BRSConfig:
    """Apply dot-notation mutations to a BRSConfig.

    Args:
        base: Base config to mutate
        mutations: Dict of "dotted.key": value pairs

    Returns:
        New BRSConfig with mutations applied
    """
    if not mutations:
        return base

    # Bucket mutations by prefix
    sym_cfg_updates: dict[str, dict[str, object]] = {}  # symbol -> field -> value
    param_override_updates: dict[str, object] = {}
    slippage_updates: dict[str, object] = {}
    top_level: dict[str, object] = {}

    for key, value in mutations.items():
        parts = key.split(".")

        if parts[0] == "symbol_configs" and len(parts) == 3:
            # symbol_configs.QQQ.adx_on → per-symbol field
            sym = parts[1]
            field_name = parts[2]
            sym_cfg_updates.setdefault(sym, {})[field_name] = value

        elif parts[0] == "param_overrides" and len(parts) == 2:
            # param_overrides.disable_s1 → merge into param_overrides
            param_override_updates[parts[1]] = value

        elif parts[0] == "slippage" and len(parts) == 2:
            # slippage.commission_per_share_etf → mutate slippage
            slippage_updates[parts[1]] = value

        else:
            # Top-level field
            field_name = parts[0]
            if hasattr(base, field_name):
                current = getattr(base, field_name)
                # bool before int (bool is subclass of int in Python)
                if isinstance(current, bool):
                    top_level[field_name] = bool(value)
                elif isinstance(current, int):
                    top_level[field_name] = int(round(value)) if isinstance(value, float) else value
                elif isinstance(current, float):
                    top_level[field_name] = float(value)
                else:
                    top_level[field_name] = value

    config = base

    # Apply symbol config mutations
    if sym_cfg_updates:
        new_sym_configs = dict(config.symbol_configs)
        for sym, updates in sym_cfg_updates.items():
            if sym in new_sym_configs:
                old_cfg = new_sym_configs[sym]
                typed_updates = {}
                for k, v in updates.items():
                    if hasattr(old_cfg, k):
                        current = getattr(old_cfg, k)
                        # bool before int (bool is subclass of int in Python)
                        if isinstance(current, bool):
                            typed_updates[k] = bool(v)
                        elif isinstance(current, int):
                            typed_updates[k] = int(round(v)) if isinstance(v, float) else v
                        elif isinstance(current, float):
                            typed_updates[k] = float(v)
                        else:
                            typed_updates[k] = v
                new_sym_configs[sym] = replace(old_cfg, **typed_updates)
        config = replace(config, symbol_configs=new_sym_configs)

    # Apply param_overrides
    if param_override_updates:
        new_overrides = dict(config.param_overrides)
        new_overrides.update(param_override_updates)
        config = replace(config, param_overrides=new_overrides)

    # Apply slippage mutations
    if slippage_updates:
        config = replace(config, slippage=replace(config.slippage, **slippage_updates))

    # Apply top-level mutations
    if top_level:
        config = replace(config, **top_level)

    return config
