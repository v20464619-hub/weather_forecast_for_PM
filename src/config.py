from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_DATA_PATH = DATA_DIR / "GlobalWeatherRepository.csv"
CLEAN_DATA_PATH = OUTPUT_DIR / "clean_weather.csv"
DAILY_DATA_PATH = OUTPUT_DIR / "daily_weather.csv"

DATE_COL = "last_updated"
TARGET_COL = "temperature_celsius"
RANDOM_STATE = 42

for d in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
