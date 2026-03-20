"""IBKR Core - Connection, pacing, mapping, reconciliation layer.

Lazy imports to allow offline testing without ib_async installed.
"""

__all__ = [
    "IBKRExecutionAdapter",
    "IBSession",
    "IBKRConfig",
    "ContractFactory",
    "ContractResolutionError",
    "BrokerOrderRef",
    "BrokerOrderStatus",
    "ExecutionReport",
    "IBContractSpec",
    "OrderStatusEvent",
    "PositionSnapshot",
    "RejectCategory",
]


def __getattr__(name: str):
    if name == "IBKRExecutionAdapter":
        from .adapters.execution_adapter import IBKRExecutionAdapter
        return IBKRExecutionAdapter
    elif name == "IBSession":
        from .client.session import IBSession
        return IBSession
    elif name == "IBKRConfig":
        from .config.loader import IBKRConfig
        return IBKRConfig
    elif name == "ContractFactory":
        from .mapping.contract_factory import ContractFactory
        return ContractFactory
    elif name == "ContractResolutionError":
        from .mapping.contract_factory import ContractResolutionError
        return ContractResolutionError
    elif name in ("BrokerOrderRef", "BrokerOrderStatus", "ExecutionReport",
                  "IBContractSpec", "OrderStatusEvent", "PositionSnapshot",
                  "RejectCategory"):
        from . import models
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
