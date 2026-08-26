"""conftest.py: 将项目根目录加入 sys.path，使 pytest 能导入 core 包。"""

import sys
from pathlib import Path

# 将项目根目录（本文件所在目录）加入 Python 路径
_PROJECT_ROOT = str(Path(__file__).parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
