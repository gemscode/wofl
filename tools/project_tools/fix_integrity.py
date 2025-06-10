#!/usr/bin/env python3
"""
RW Agent Integrity Fixer
"""
import os
import sys
from pathlib import Path
from textwrap import dedent

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

REQUIRED_DIRS = [
    "bin",
    "src",
    "src/agents",
    "src/utils",
    "src/templates/docker"
]

def ensure_directory(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    except Exception as e:
        print(f"Error creating {path}: {str(e)}")

def create_file_with_content(path: Path, content: str = ""):
    if path.exists():
        if path.stat().st_size == 0:
            print(f"Populating empty file: {path}")
        else:
            return  # Don't overwrite existing content
    else:
        print(f"Creating file: {path}")
    
    with open(path, "w") as f:
        f.write(content)

def fix_directory_structure(base_path: Path):
    # Create required directories
    for dir_path in REQUIRED_DIRS:
        ensure_directory(base_path / dir_path)

    # Create required files with initial content
    for category in REQUIRED_FILES.values():
        for file_path in category:
            full_path = base_path / file_path
            
            # Special handling for different file types
            if file_path.endswith("__init__.py"):
                content = "# Package initialization\n"
            elif "Dockerfile.j2" in file_path:
                content = dedent("""\
                    # Base Docker template
                    FROM python:3.9-slim
                    WORKDIR /app
                    COPY . .
                    RUN pip install -r requirements.txt
                    CMD ["python", "app.py"]
                """)
            elif file_path == "src/app.py":
                content = dedent("""\
                    # Main application entry point
                    from src.utils.config import load_config

                    def main():
                        print("R&W Agent Service")

                    if __name__ == "__main__":
                        main()
                """)
            elif file_path == ".env":
                content = dedent("""\
                    # Environment Configuration
                    ELASTICSEARCH_URL=http://localhost:9200
                    CASSANDRA_HOST=127.0.0.1
                    CASSANDRA_KEYSPACE=rw_agent
                    ANTHROPIC_API_KEY=your_key_here
                """)
            else:
                content = f"# Placeholder file: {file_path}\n"
            
            create_file_with_content(full_path, content)

def main():
    base_path = Path(os.getcwd())
    print("🛠️  Fixing project integrity...\n")
    
    try:
        fix_directory_structure(base_path)
        print("\n✅ Integrity fixes applied successfully!")
        print("Note: Review placeholder files and customize as needed.")
    except Exception as e:
        print(f"❌ Critical error during repair: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

