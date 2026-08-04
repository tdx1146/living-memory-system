"""
活体记忆系统 (Living Memory System) - 包安装配置

安装: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name='living-memory-system',
    version='0.1.0',
    description='活体记忆系统 - 外挂于主LLM的海马体记忆层',
    long_description=(
        '一个外挂于主LLM的"海马体"记忆层。'
        '主LLM负责思考（黑箱API），本系统负责让思考被记住、被遗忘、被重新激活。'
        '学习规则从自由能原理（FEP）涌现，不需要预训练、不需要反向传播。'
    ),
    long_description_content_type='text/plain',
    author='dandan + AI',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*', 'docs']),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0',
        'numpy>=1.21',
        'openai>=1.0',
        'fastapi>=0.100',
        'uvicorn[standard]>=0.20',
        'sentence-transformers>=2.2',
        'mcp>=1.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-asyncio>=0.21',
            'httpx>=0.24',
            'pytest-cov>=4.0',
            'detect-secrets>=1.4',
        ],
    },
    entry_points={
        'console_scripts': [
            'lms=runtime.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
