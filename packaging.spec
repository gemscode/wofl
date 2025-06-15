# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

base_path = os.getcwd()

data_files = [
    # Root files
    ('.env_sample', '.'),
    ('LICENSE', '.'),
    ('README.md', '.'),
    ('requirements.in', '.'),
    ('requirements.txt', '.'),
    ('install_libev.sh', '.'),

    # Cassandra
    ('src/deployments/cassandra/create_tables.cql', 'src/deployments/cassandra'),
    ('src/deployments/cassandra/docker-compose.yml', 'src/deployments/cassandra'),
    ('src/deployments/cassandra/init.sh', 'src/deployments/cassandra'),

    # Elasticsearch
    ('src/deployments/elasticsearch/docker-compose.yml', 'src/deployments/elasticsearch'),
    ('src/deployments/elasticsearch/init.sh', 'src/deployments/elasticsearch'),

    # Kafka
    ('src/deployments/kafka/docker-compose.yml', 'src/deployments/kafka'),
    ('src/deployments/kafka/Dockerfile.setup', 'src/deployments/kafka'),
    ('src/deployments/kafka/init.sh', 'src/deployments/kafka'),
    ('src/deployments/kafka/producer.py', 'src/deployments/kafka'),
    ('src/deployments/kafka/requirements.txt', 'src/deployments/kafka'),

    # k3s
    ('src/deployments/k3s/init.sh', 'src/deployments/k3s'),

    # Redis
    ('src/deployments/redis/docker-compose.yml', 'src/deployments/redis'),
    ('src/deployments/redis/init.sh', 'src/deployments/redis'),
    ('src/deployments/redis/requirements.txt', 'src/deployments/redis'),

    # Framework components
    ('src/framework', 'src/framework'),
    ('src/rw_agent', 'src/rw_agent'),
    ('src/tools', 'src/tools'),
    ('src/projects', 'src/projects'),

    # CLI
    ('src/rw_agent', 'src/rw_agent'),


    # System dependencies
    ('lib/libev-4.33', 'lib/libev-4.33'),
    ('lib/libev-install', 'lib/libev-install'),

    # Virtual environment
    ('core_env', 'core_env')
]

hiddenimports = collect_submodules('cassandra') + collect_submodules('elasticsearch') + ['rw_agent', 'rw_agent.cli', 'rw_agent.cli.commands']

a = Analysis(
    ['launcher.py'],
    pathex=[base_path],
    binaries=[],
    datas=data_files,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='WolfX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)

