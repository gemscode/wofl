# WolfXE Trading Simulator

## Complete Setup & Build Steps

**Here is the full list of steps to set up your environment and run WolfXE Trading Simulator:**

1. Navigate to the wolfxe branch directory
2. Create and activate a Python virtual environment
3. Install all required dependencies
4. Configure Redis connection
5. Prepare simulation data for trading symbols
6. Run the trading simulator

## Step-by-Step Breakdown

### 1. Navigate to Project Directory

```bash
cd wolfxe
```

### 2. Create and Activate Virtual Environment

```bash
# Automatic setup (recommended)
./setenv.sh

# Manual setup
python3 -m venv wolfxe_env
source wolfxe_env/bin/activate
pip install -r requirements.txt
```

### 3. Install Dependencies

```bash
# If using manual setup
pip install -r requirements.txt
```

### 4. Configure Redis Connection

Create `.redis_passwd` file in project root:
```bash
echo "your_redis_password" > .redis_passwd
```

### 5. Prepare Simulation Data

```bash
# Navigate to simulator directory
cd simulator

# Prepare data for a specific symbol (downloads 60 days, trains on 40, tests on 20)
python prepare.py AAPL
python prepare.py GERN
```

### 6. Run Trading Simulator

```bash
# Activate environment (if not already active)
source ../wolfxe_env/bin/activate

# Run simulation
python main.py --symbol AAPL --budget 25000 --speed 5.0
```

## Available Commands

### Environment Management
```bash
./setenv.sh              # Setup environment and dependencies
./activate.sh            # Quick environment activation
```

### Data Preparation
```bash
cd simulator
python prepare.py SYMBOL              # Prepare symbol data
python prepare.py SYMBOL --force      # Force re-download
```

### Simulation Options
```bash
cd simulator
python main.py --symbol AAPL --budget 25000 --speed 5.0
python main.py --symbol GERN --budget 25000 --speed 5.0 --aggressive
```

## Command Line Arguments

- `--symbol`: Stock symbol (AAPL, GERN, TSLA, etc.)
- `--budget`: Trading budget in USD (default: 25000)
- `--speed`: Simulation speed multiplier (default: 10.0)
- `--aggressive`: Enable aggressive trading mode
- `--redis-host`: Redis server hostname (default: trader.wolfx0.com)
- `--redis-port`: Redis server port (default: 6379)

## Post-Simulation Options

After simulation completion, you can:
1. Exit and use current training for live trading
2. Retrain with aggressive approach
3. View detailed trade log
4. Configure window range and rerun simulation

## Supported Symbols

The simulator supports **any publicly traded stock symbol** available on Yahoo Finance. The system will automatically:
- Download 60 days of 5-minute intraday data
- Train on the first 40 days
- Test on the remaining 20 days

### Example Symbols
- AAPL (Apple Inc.)
- GERN (Geron Corporation) 
- TSLA (Tesla Inc.)
- NVDA (NVIDIA Corporation)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- GOOGL (Alphabet Inc.)
- META (Meta Platforms Inc.)

### Adding New Symbols
```bash
# Prepare any symbol for simulation
cd simulator
python prepare.py SYMBOL_NAME

# Examples
python prepare.py AMZN
python prepare.py GOOGL
python prepare.py META
python prepare.py SPY
```

The system will automatically optimize window ranges for each symbol based on historical performance patterns.

## Project Structure

```
wolfxe/
├── simulator/           # Trading simulation engine
├── production/          # Production trading system
├── agents/             # Trading agents
├── config/             # Configuration files
├── models/             # Trained model storage
├── shared/             # Shared utilities
├── tests/              # Test suites
├── docs/               # Documentation
├── results/            # Simulation results
├── wolfxe_env/         # Virtual environment
├── setenv.sh           # Environment setup script
├── activate.sh         # Quick activation script
└── requirements.txt    # Python dependencies
```

## Requirements

- Python 3.10+
- Redis server access
- Internet connection for data download
- 2GB+ available disk space

## Troubleshooting

- **Redis connection failed**: Check `.redis_passwd` file and network connectivity
- **No simulation data**: Run `python prepare.py SYMBOL` first
- **Import errors**: Ensure virtual environment is activated and you're in the simulator directory
- **Permission denied**: Run `chmod +x setenv.sh activate.sh`
- **Module not found**: Make sure you're running commands from the correct directory (simulator/ for simulation commands)

## License

MIT License
Copyright (c) 2025 WolfXE Trading Simulator
