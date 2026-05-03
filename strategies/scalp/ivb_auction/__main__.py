from __future__ import annotations

import json

from .core.serializers import snapshot_state
from .core.state import IvbAuctionCoreState


def main() -> int:
    print(json.dumps(snapshot_state(IvbAuctionCoreState()), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

