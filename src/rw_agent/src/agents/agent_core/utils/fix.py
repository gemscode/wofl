# src/agents/agent_core/utils/fix.py
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
    """Create directory if missing"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    except Exception as e:
        print(f"Error creating {path}: {str(e)}")
        raise

def create_file_with_content(path: Path, content: str = ""):
    """Create or repair file with safe content"""
    if path.exists():
        if path.stat().st_size == 0:
            print(f"Populating empty file: {path}")
        else:
            return  # Preserve existing content
    else:
        print(f"Creating file: {path}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        f.write(content)

def get_file_template(file_path: str) -> str:
    """Return appropriate template content for different file types"""
    if file_path.endswith("__init__.py"):
        return "# Package initialization\n"
    
    if "Dockerfile.j2" in file_path:
        return dedent("""\
            # Base Docker template
            FROM python:3.9-slim
            WORKDIR /app
            COPY . .
            RUN pip install -r requirements.txt
            CMD ["python", "app.py"]
        """)
    
    if file_path == "src/app.py":
        return dedent("""\
            # Main application entry point
            from src.utils.config import load_config

            def main():
                print("R&W Agent Service")

            if __name__ == "__main__":
                main()
        """)
    
    if file_path == ".env":
        return dedent("""\
            # Environment Configuration
            ELASTICSEARCH_URL=http://localhost:9200
            CASSANDRA_HOST=127.0.0.1
            CASSANDRA_KEYSPACE=rw_agent
            ANTHROPIC_API_KEY=your_key_here
        """)
    
    return f"# Placeholder file: {file_path}\n"

def fix_project(base_path: Path):
    """Main repair entry point"""
    # Create directories
    for dir_path in REQUIRED_DIRS:
        ensure_directory(base_path / dir_path)

    # Create files
    for category in REQUIRED_FILES.values():
        for file_path in category:
            full_path = base_path / file_path
            content = get_file_template(file_path)
            create_file_with_content(full_path, content)

# Standalone execution (for debugging)
if __name__ == "__main__":
    base_path = Path(os.getcwd())
    print("🛠️  Fixing project integrity...")
    
    try:
        fix_project(base_path)
        print("\n✅ Integrity fixes applied!")
        print("Note: Review and customize generated files.")
    except Exception as e:
        print(f"❌ Repair failed: {str(e)}")
        sys.exit(1)

