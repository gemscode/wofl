from setuptools import setup, find_packages

setup(
    name="rwagent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'rwagent=rw_agent.cli.main:cli',
        ],
    },
    include_package_data=True,
)

