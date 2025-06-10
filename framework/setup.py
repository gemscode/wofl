from setuptools import setup, find_packages

setup(
    name="rwagent",
    version="0.1.0",
    packages=find_packages(include=["cli", "cli.*", "utils", "utils.*"]),
    entry_points={
        "console_scripts": [
            "rwagent=cli.main:cli"
        ]
    },
    install_requires=[],  # dependencies handled in core_requirements
    include_package_data=True,
    zip_safe=False,
)

