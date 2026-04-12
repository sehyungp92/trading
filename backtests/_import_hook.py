"""MetaPathFinder that redirects legacy backtest imports to monorepo locations.

Each family (momentum, swing) defines an alias map like::

    {"strategy": "strategies.momentum.helix_v40",
     "strategy_2": "strategies.momentum.nqdtc",
     "backtest": "research.backtests.momentum"}

The finder intercepts any import whose top-level package matches an alias key,
rewrites the module name, imports the real module, and registers it under
*both* the old and new names in ``sys.modules`` so subsequent imports resolve
instantly without hitting the finder again.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from importlib.abc import MetaPathFinder


class _AliasRedirector(MetaPathFinder):
    """Redirect old-style imports to their monorepo equivalents."""

    def __init__(self, aliases: dict[str, str]) -> None:
        # Sort longest-prefix-first so "strategy_2" matches before "strategy"
        self._aliases = dict(
            sorted(aliases.items(), key=lambda kv: -len(kv[0]))
        )
        self._active = False  # recursion guard

    def find_spec(self, fullname: str, path=None, target=None):
        """Return a ModuleSpec that loads the real module for aliased names."""
        if self._active:
            return None

        for old_prefix in self._aliases:
            if fullname == old_prefix or fullname.startswith(old_prefix + "."):
                real_name = self._rewrite(fullname)

                # Use recursion guard instead of removing from meta_path
                self._active = True
                try:
                    real_spec = importlib.util.find_spec(real_name)
                finally:
                    self._active = False

                if real_spec is None:
                    return None

                # Create a spec under the aliased name that uses the real
                # module's loader, but also registers the alias in sys.modules.
                loader = _AliasLoader(real_name, real_spec, fullname, self)
                spec = importlib.machinery.ModuleSpec(
                    fullname,
                    loader,
                    origin=real_spec.origin,
                    is_package=real_spec.submodule_search_locations is not None,
                )
                if real_spec.submodule_search_locations is not None:
                    spec.submodule_search_locations = list(
                        real_spec.submodule_search_locations
                    )
                return spec
        return None

    def _rewrite(self, fullname: str) -> str:
        for old_prefix, new_prefix in self._aliases.items():
            if fullname == old_prefix:
                return new_prefix
            if fullname.startswith(old_prefix + "."):
                suffix = fullname[len(old_prefix):]
                return new_prefix + suffix
        return fullname


class _AliasLoader:
    """Loader that imports the real module and registers the alias."""

    def __init__(self, real_name, real_spec, alias_name, redirector):
        self._real_name = real_name
        self._real_spec = real_spec
        self._alias_name = alias_name
        self._redirector = redirector

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        # Import the real module.  Do NOT set _active here — nested imports
        # within the real module (e.g. ``from strategy.config import X``
        # inside an engine file) must still be intercepted by the finder.
        # The recursion guard in find_spec() is sufficient to prevent loops.
        real_mod = importlib.import_module(self._real_name)

        # Register both alias and real names to the SAME module object.
        # This is critical: strategy code uses relative imports (from . import config)
        # which resolve to the real module, while engine code uses aliased imports
        # (from strategy_2 import config).  If these are different objects, setattr
        # on one (e.g. param_overrides patching) won't affect the other.
        sys.modules[self._real_name] = real_mod
        sys.modules[self._alias_name] = real_mod

        # Register ancestor aliases so "from strategy.config import X" works
        parts = self._alias_name.split(".")
        for i in range(1, len(parts)):
            ancestor = ".".join(parts[:i])
            if ancestor not in sys.modules:
                real_ancestor = self._redirector._rewrite(ancestor)
                if real_ancestor in sys.modules:
                    sys.modules[ancestor] = sys.modules[real_ancestor]


def install(aliases: dict[str, str]) -> None:
    """Install the alias redirector if not already present."""
    for finder in sys.meta_path:
        if isinstance(finder, _AliasRedirector):
            if finder._aliases == aliases:
                return  # already installed (exact match)
            # Detect conflicting overlaps — two families mapping the same
            # key (e.g. "strategy") to different targets in one process.
            for key in aliases:
                if key in finder._aliases and finder._aliases[key] != aliases[key]:
                    raise RuntimeError(
                        f"Conflicting import alias for {key!r}: "
                        f"existing={finder._aliases[key]!r}, "
                        f"new={aliases[key]!r}. "
                        f"Only one backtest family can be active per process."
                    )
    sys.meta_path.insert(0, _AliasRedirector(aliases))
