"""持久化层

包含吸引子景观快照和断点续传功能。
"""

from persistence.snapshot import Snapshot, SNAPSHOT_VERSION
from persistence.recovery import Recovery

__all__ = ["Snapshot", "SNAPSHOT_VERSION", "Recovery"]
