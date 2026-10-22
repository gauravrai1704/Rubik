"""All seven methods (six from the technical basis, plus 5C). Each sub-package registers its solver under a short name:

=================  ====================================  ================================
name               method                                owner
=================  ====================================  ================================
m1_idastar         IDA* + pattern databases              Shree (implemented by Gaurav)
m2_reduction       Reduction + Kociemba two-phase        Sunaina (implemented by Gaurav)
m3_demaine         Demaine parallel cluster algorithm    Kashvi (implemented by Gaurav)
m4_cayleypy        CayleyPy reproduction                 Shaurya (implemented by Gaurav)
m5a_tokens         Cluster-token model, one for all N    Sharanya (implemented by Gaurav)
m5b_stages         Learned search inside stages          Gaurav
m5c_schedule       Net-gain scheduling of bulk commutators  Gaurav
=================  ====================================  ================================
"""
from .base import REGISTRY, Budget, Solver, SolveResult, get_solver, register

_LOADED = False


def load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    import importlib

    for mod in ("m1_idastar", "m2_reduction", "m3_demaine", "m4_cayleypy", "m5a_tokens", "m5b_stages", "m5c_schedule"):
        importlib.import_module(f"{__name__}.{mod}")
    _LOADED = True


__all__ = ["Budget", "Solver", "SolveResult", "register", "get_solver", "REGISTRY", "load_all"]
