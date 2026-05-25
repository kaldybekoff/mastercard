"""
config.py — все константы и пути проекта MDQ 2026
Импортируй из любого места: from src.config import DATA_DIR, B2B_MCC_CODES
"""

from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CONFIGS_DIR = ROOT_DIR / "configs"

# ─── Файлы данных ─────────────────────────────────────────────────────────────

BUSINESS_CARDS_PATH = RAW_DIR / "business_cards_MDQ.parquet"
CONSUMER_CARDS_PATH = RAW_DIR / "consumer_cards_MDQ.parquet"
MERCHANTS_PATH = RAW_DIR / "merchants_reference.parquet"

FEATURE_MATRIX_PATH = PROCESSED_DIR / "feature_matrix.parquet"

# ─── Воспроизводимость ────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ─── Train/test split ─────────────────────────────────────────────────────────

TRAIN_SIZE = 0.8   # 80% business cards на обучение
TEST_SIZE = 0.2    # 20% business cards на holdout

# ─── PU-Bagging ───────────────────────────────────────────────────────────────

PU_N_ITERATIONS = 10       # Количество итераций bagging
PU_SAMPLE_SIZE = 25_000    # Размер unlabeled sample = размер позитивного класса

# ─── MCC классификация ────────────────────────────────────────────────────────

# B2B MCC — явный бизнес-сигнал
B2B_MCC_CODES = {
    7372,  # Software / SaaS
    5045,  # Computers & IT equipment
    7311,  # Advertising services (Google Ads, Meta Ads)
    2741,  # Printing & publishing
    5300,  # Wholesale clubs / Оптовая торговля
    5111,  # Stationery / Office supplies
    4214,  # Freight / Logistics
    7392,  # Consulting / Management services
    5065,  # Electrical equipment wholesale
    5040,  # Professional equipment
    5085,  # Industrial supplies
    5169,  # Chemicals wholesale
    7389,  # Business services
    8049,  # Offices & clinics (может быть бизнес)
    4215,  # Courier services
    # Кандидаты, найденные по EDA (топ-20 MCC бар-чарт, §3.3 в 01_eda.ipynb):
    # эти коды у business доминируют, у consumer почти отсутствуют.
    5968,  # Direct marketing / subscription services (Zoom, Shopify, Slack)
    4816,  # Computer network / hosting services (Cloudflare, DigitalOcean, GoDaddy, Hetzner)
    7399,  # Business services NEC
    8931,  # Accounting / auditing / bookkeeping
    7379,  # Computer services / repair
    5046,  # Commercial equipment NEC
}

# Consumer MCC — потребительский сигнал
CONSUMER_MCC_CODES = {
    5812,  # Restaurants / Eateries
    5814,  # Fast food
    7832,  # Movie theaters
    7995,  # Entertainment
    5411,  # Grocery stores / Supermarkets
    5311,  # Department stores
    5912,  # Drug stores / Pharmacy
    7011,  # Hotels
    4111,  # Local transit
    5942,  # Book stores
    7922,  # Ticketing
}

# Mixed MCC — могут быть и личные и бизнес
MIXED_MCC_CODES = {
    5541,  # Gas stations (АЗС)
    4814,  # Telecom / Mobile
    4121,  # Taxis / Ridesharing
    5499,  # Misc food stores
    5200,  # Hardware stores
    7349,  # Cleaning services
}

# ─── Временные паттерны ───────────────────────────────────────────────────────

BUSINESS_HOURS_START = 9   # Рабочий день с 9:00
BUSINESS_HOURS_END = 18    # Рабочий день до 18:00
EVENING_START = 18
EVENING_END = 23
NIGHT_START = 23
NIGHT_END = 6
MORNING_START = 6
MORNING_END = 9

BUSINESS_DAYS = {1, 2, 3, 4, 5}  # Пн-Пт (Polars dt.weekday(): 1=Mon, 7=Sun)
WEEKEND_DAYS = {6, 7}             # Сб-Вс

# ─── Порог скоринга (threshold) ───────────────────────────────────────────────
# Управляется слайдером в дашборде. Здесь — дефолт для ноутбука.

DEFAULT_THRESHOLD = 0.7

# ─── Segmentation ─────────────────────────────────────────────────────────────

N_SEGMENTS = 5  # Количество кластеров для KMeans

SEGMENT_NAMES = {
    0: "E-commerce sellers",
    1: "Offline retail",
    2: "Service providers",
    3: "Restauranteurs",
    4: "Digital / IT",
}

# ─── ROI assumptions (для бизнес-расчёта) ────────────────────────────────────

HIDDEN_ENTREPRENEUR_SHARE = 0.06   # 6% от consumer = скрытые предприниматели
CONVERSION_RATE = 0.15             # 15% конверсия при активном маркетинге
LTV_PER_CLIENT_KZT = 200_000      # 200K₸/год на конвертированного клиента
