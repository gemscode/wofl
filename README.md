```markdown
# WolfX: R&W AI Companion Mac Executable

---

## Complete Setup & Build Steps

**Here is the full list of steps to set up your environment, build, and test WolfX:**

1. Create and activate a clean Python 3.12 virtual environment
2. Install required build tools (`pip-tools`, `pyinstaller`)
3. Ensure PyInstaller is in your PATH (if not using a virtualenv)
4. Verify PyInstaller installation
5. Install all Python dependencies
6. Build the WolfX executable
7. Test the executable in a new directory
8. Clean and rebuild (if needed)

---

## Step-by-Step Breakdown

### 1. Create and Activate a Clean Python 3.12 Virtual Environment

```
cd /path/to/your/project/root
python3.12 -m venv core_env
source core_env/bin/activate
```

---

### 2. Install Required Build Tools

```
pip install pip-tools
pip install pyinstaller
```

---

### 3. Ensure PyInstaller is in Your PATH (if not using a virtualenv)

If you installed PyInstaller with `--user` (not in a venv), add this to your shell:
```
export PATH="$HOME/.local/bin:$PATH"
```
Add it to your `~/.zshrc` for persistence.

---

### 4. Verify PyInstaller Installation

```
pyinstaller --version
# Should show a version number (e.g., 6.14.1)
```

---

### 5. Install All Python Dependencies

```
pip-compile requirements.in
pip install -r requirements.txt
```

---

### 6. Build the WolfX Executable

```
pyinstaller packaging.spec --clean
```

---

### 7. Test the Executable in a New Directory

```
# Copy the executable to a clean directory for testing
cp dist/WolfX ~/test_wolfx/
cd ~/test_wolfx

chmod +x WolfX
./WolfX
```

---

### 8. Clean and Rebuild (if needed)

```
rm -rf build dist
pyinstaller packaging.spec --clean
cd dist && ./WolfX
```

---

## Usage Notes

- The executable will create a `.env` file in your working directory if one does not exist.
- All required Docker services (Cassandra, Elasticsearch, Kafka, Redis, k3s) will be initialized automatically.
- You can upload the `WolfX` executable to a GitHub release for easy sharing.
- For more details, see the [Releases](./releases) page.

---

## Troubleshooting

- **PyInstaller not found:**  
  Make sure you installed it in the right environment and your `PATH` includes the correct `bin` directory.
- **NumPy/Elasticsearch errors:**  
  Pin `numpy<2.0` in your requirements if you see compatibility issues.
- **Missing `core_env`:**  
  Ensure your virtual environment exists in the project root as `core_env`.

---

## License

MIT License  
Copyright (c) 2025 R&W WolfX

See [LICENSE](./LICENSE) for more information.
```
