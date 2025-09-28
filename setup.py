#!/usr/bin/env python3
"""
Setup script for Magic Terminal - A comprehensive AI-powered terminal assistant
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Magic Terminal - A comprehensive AI-powered terminal assistant"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "requests>=2.28.0",
            "psutil>=5.9.0",
            "jsonschema>=4.0.0"
        ]

setup(
    name="magic-terminal-cli",
    version="1.0.0",
    author="Magic Terminal Team",
    author_email="support@magicterminal.dev",
    description="A magical terminal assistant that understands natural language and makes command-line operations effortless",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Yogesh-developer/magic-terminal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Shells",
        "Topic :: System :: System Shells",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "magic-terminal=ai_terminal.main:main",
            "magic=ai_terminal.main:main",  # Short alias
            "mt=ai_terminal.main:main",     # Ultra short alias
        ],
    },
    include_package_data=True,
    package_data={
        "ai_terminal": ["templates/*", "config/*"],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
        ],
        "full": [
            "colorama>=0.4.4",
            "rich>=12.0.0",
            "prompt-toolkit>=3.0.0",
        ],
    },
    keywords="terminal, magic, assistant, automation, cli, shell, natural-language",
    project_urls={
        "Bug Reports": "https://github.com/Yogesh-developer/magic-terminal/issues",
        "Source": "https://github.com/Yogesh-developer/magic-terminal",
        "Documentation": "https://github.com/Yogesh-developer/magic-terminal/wiki",
    },
)
