import os
import subprocess
import sys
from pathlib import Path

def get_base_path():
    # For PyInstaller packaged execution
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    # For development/direct execution
    return os.path.dirname(os.path.abspath(__file__))

def run_shell_script(script_path):
    env = os.environ.copy()
    if getattr(sys, 'frozen', False):
        env["MEIPASS"] = sys._MEIPASS  # Pass temp dir to subprocess
    # Ensure script is executable
    os.chmod(script_path, 0o755)
    result = subprocess.run(
        [script_path],
        env=env,
        shell=False,
        check=True
    )
    return result.returncode

def main():
    base_path = get_base_path()

    # 1. Ensure .env exists in the CWD (project root)
    env_sample = Path(base_path) / ".env_sample"
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        if env_sample.exists():
            env_file.write_text(env_sample.read_text())
            print("Created .env from .env_sample")
        else:
            print("❌ .env_sample not found. Cannot create .env.")
            sys.exit(1)
    else:
        print(".env already exists, skipping creation.")

    # 2. Install libev and Cassandra driver (if needed)
    install_libev_path = os.path.join(base_path, "install_libev.sh")
    if os.path.exists(install_libev_path):
        print("Running install_libev.sh...")
        run_shell_script(install_libev_path)
    else:
        print(f"❌ {install_libev_path} not found.")
        sys.exit(1)

    # 3. Run main initialization script
    init_sh_path = os.path.join(base_path, "src", "framework", "init.sh")
    if os.path.exists(init_sh_path):
        print("Running framework/src/framework/init.sh...")
        run_shell_script(init_sh_path)
    else:
        print(f"❌ {init_sh_path} not found.")
        sys.exit(1)

    print("\n✅ All steps completed successfully. You can now use the CLI and services.")

if __name__ == "__main__":
    main()

