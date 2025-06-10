#!/usr/bin/env python3
"""
RW Agent Integrity Checker (Improved Tree Format)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
SCRIPT_PATH = Path(__file__).resolve()
FRAMEWORK_DIR = SCRIPT_PATH.parents[1]
ROOT_DIR = SCRIPT_PATH.parents[3]  # Corrected to always resolve to true root
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

REQUIRED_FILES = {
    "root": [
        "README.md",
        "requirements.txt",
        "setup.py",
        "bin/register_agent.py",
        "bin/query_agent_info.py"
    ],
    "src": [
        "src/__init__.py",
        "src/app.py",
        "src/templates/docker/Dockerfile.j2"
    ],
    "agents": [
        "src/agents/agent_core/__init__.py",
        "src/agents/agent_core/core.py",
        "src/agents/agent_ai/__init__.py",
        "src/agents/agent_ai/core.py",
        "src/agents/agent_storage/__init__.py",
        "src/agents/agent_storage/core.py"
    ],
    "utils": [
        "src/utils/__init__.py",
        "src/utils/cassandra_manager.py",
        "src/utils/elasticsearch_manager.py"
    ]
}

REQUIRED_DIRS = [
    "bin", "src", "src/agents", "src/utils", "src/templates/docker"
]

REQUIRED_ENV_VARS = [
    "ELASTICSEARCH_URL", "CASSANDRA_HOST", "CASSANDRA_KEYSPACE"
]

def check_directories(base: Path):
    results = {"errors": [], "warnings": [], "tree": []}
    for d in REQUIRED_DIRS:
        full = base / d
        if not full.exists():
            results["errors"].append(f"Missing directory: {d}")
        elif not any(full.iterdir()):
            results["warnings"].append(f"Empty directory: {d}")
        results["tree"].append(f"📁 {d}")
    return results

def check_files(base: Path):
    results = {"errors": [], "warnings": [], "tree": []}
    for section, files in REQUIRED_FILES.items():
        for f in files:
            full = base / f
            if not full.exists():
                results["errors"].append(f"Missing file: {f}")
            elif full.stat().st_size == 0:
                results["warnings"].append(f"Empty file: {f}")
            results["tree"].append(f"  📄 {f}")
    return results

def check_env():
    results = {"errors": [], "warnings": []}
    if not ENV_PATH.exists():
        results["warnings"].append("Missing .env file at root directory")
        return results

    with open(ENV_PATH) as f:
        lines = f.readlines()
    keys = {line.split("=")[0].strip() for line in lines if "=" in line}
    for var in REQUIRED_ENV_VARS:
        if var not in keys:
            results["warnings"].append(f"Missing .env variable: {var}")
    return results

def main():
    base = FRAMEWORK_DIR
    print(f"🔍 Running integrity checks in: {base}\n")

    errors, warnings, tree_output = [], [], []
    for check in [check_directories, check_files]:
        result = check(base)
        errors.extend(result["errors"])
        warnings.extend(result["warnings"])
        tree_output.extend(result["tree"])

    env_result = check_env()
    warnings.extend(env_result["warnings"])  # Only consider as warning

    print("📦 Project Structure:")
    for line in tree_output:
        print(line)

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("\n✅ All integrity checks passed!")

    print("\n📊 Summary:")
    print(f"- Errors: {len(errors)}")
    print(f"- Warnings: {len(warnings)}")
    print(f"- Files checked: {sum(len(f) for f in REQUIRED_FILES.values())}")
    print(f"- Directories checked: {len(REQUIRED_DIRS)}")

if __name__ == "__main__":
    main()

