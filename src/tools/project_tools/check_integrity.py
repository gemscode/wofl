#!/usr/bin/env python3
"""
RW Agent Integrity Checker
"""
import os
from pathlib import Path
from tabulate import tabulate

REQUIRED_FILES = {
    "root": [
        "README.md",
        "requirements.txt",
        "setup.py",
        ".env",
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

def check_directory_structure(base_path: Path) -> dict:
    results = {"errors": [], "warnings": []}
    
    # Check required directories
    required_dirs = [
        "bin",
        "src",
        "src/agents",
        "src/utils",
        "src/templates/docker"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            results["errors"].append(f"Missing directory: {dir_path}")
        elif not any(full_path.iterdir()):
            results["warnings"].append(f"Empty directory: {dir_path}")

    # Check required files
    for category, files in REQUIRED_FILES.items():
        for file_path in files:
            full_path = base_path / file_path
            if not full_path.exists():
                results["errors"].append(f"Missing file: {file_path}")
            elif full_path.stat().st_size == 0:
                results["warnings"].append(f"Empty file: {file_path}")

    return results

def check_env_file(base_path: Path) -> dict:
    env_path = base_path / ".env"
    results = {"errors": [], "warnings": []}
    
    required_env_vars = [
        "ELASTICSEARCH_URL",
        "CASSANDRA_HOST",
        "CASSANDRA_KEYSPACE"
    ]
    
    if not env_path.exists():
        results["errors"].append("Missing .env file")
        return results
        
    with open(env_path) as f:
        content = f.read()
        
    existing_vars = [line.split("=")[0] for line in content.splitlines() if line.strip()]
    
    for var in required_env_vars:
        if var not in existing_vars:
            results["warnings"].append(f"Missing .env variable: {var}")
            
    return results

def main():
    base_path = Path(os.getcwd())
    print(f"🔍 Checking project integrity at: {base_path}\n")
    
    # Run checks
    structure_results = check_directory_structure(base_path)
    env_results = check_env_file(base_path)
    
    # Combine results
    all_errors = structure_results["errors"] + env_results["errors"]
    all_warnings = structure_results["warnings"] + env_results["warnings"]
    
    # Create report
    report = []
    for err in all_errors:
        report.append(("Error", err))
    for warn in all_warnings:
        report.append(("Warning", warn))
        
    # Print results
    if report:
        print(tabulate(report, headers=["Type", "Message"], tablefmt="fancy_grid"))
    else:
        print("✅ All integrity checks passed!")
        
    # Statistics
    print(f"\nSummary:")
    print(f"- Total errors: {len(all_errors)}")
    print(f"- Total warnings: {len(all_warnings)}")
    print(f"- Files/directories checked: {sum(len(v) for v in REQUIRED_FILES.values()) + 4}")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
        main()
    except ImportError:
        print("Error: Install tabulate first - pip install tabulate")

