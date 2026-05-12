"""
Compatibility shim — re-exports from app.server.

The classes BiologicalSystemSimulator, DrugProperties, and BiologicalParameters
were originally defined in this module and later moved to app/server.py.
Importing from here preserves backwards-compatibility for tests and external scripts.
"""
from app.server import BiologicalSystemSimulator, DrugProperties, BiologicalParameters  # noqa: F401
