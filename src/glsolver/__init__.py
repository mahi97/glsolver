"""glsolver: exact polynomial-time Győri–Lovász partitioning (arXiv 2608.30945)."""
from glsolver.api import GLResult, choose_algorithm, core_available, glpartition, partition
from glsolver.instance import Instance, from_networkx, make_instance
from glsolver.verify import VerificationReport, verify_instance_parts, verify_partition

__all__ = [
    "GLResult", "Instance", "VerificationReport", "choose_algorithm", "core_available",
    "from_networkx", "glpartition", "make_instance", "partition", "verify_instance_parts",
    "verify_partition",
]
__version__ = "0.1.0"
