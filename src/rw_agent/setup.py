from setuptools import setup, find_packages

setup(
    name="rwagent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'rwagent=agents.agent_core.cli.main:cli',
        ],
    },
    install_requires=[
        'click>=8.0',
        'python-dotenv>=0.19',
        'cassandra-driver>=3.28',
        'elasticsearch>=8.5',
        'tabulate>=0.9'
    ],
    include_package_data=True,
)

