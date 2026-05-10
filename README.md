

## Getting Started

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

# Kill the app
 lsof -ti :8082 -ti :8765 | xargs kill -9 2>/dev/null && pkill -f "python3 main.py" 2>/dev/null; echo "done"