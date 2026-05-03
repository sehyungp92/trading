from __future__ import annotations

import json

from .core.state import Po3ReversalCoreState
from .core.serializers import snapshot_state


def main() -> int:
    print(json.dumps(snapshot_state(Po3ReversalCoreState()), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

