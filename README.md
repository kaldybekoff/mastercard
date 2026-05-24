# Mastercard Data Quest 2026 — Hidden Entrepreneur Detection

**Хакатон:** AIESEC × Mastercard Data Quest  
**Период:** 23 мая — 30 мая 2026  
**Призовой фонд:** 2 500 000 ₸  
**Город:** Алматы

---

## Задача

Разработать ML-алгоритм, который по транзакционному поведению обнаруживает скрытую коммерческую активность среди клиентов-физлиц (consumer cards).

**Сдаваемые материалы:**
1. Jupyter Notebook — воспроизводимый, запускается сверху вниз без ошибок
2. PowerPoint презентация — методология, фичи, рекомендации

---

## Быстрый старт

```bash
# 1. Клонировать репо
git clone <repo-url>
cd mdq-hackathon

# 2. Создать conda environment
conda create -n mdq python=3.11
conda activate mdq

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Положить данные (не в git!)
# data/raw/business_cards_MDQ.parquet
# data/raw/consumer_cards_MDQ.parquet
# data/raw/merchants_reference.parquet

# 5. Запустить финальный ноутбук
jupyter lab notebooks/FINAL_pipeline.ipynb
```

---

## Структура проекта

```
mdq-hackathon/
├── data/
│   ├── raw/                    # Исходные parquet файлы (не в git)
│   ├── interim/                # Промежуточные файлы
│   └── processed/              # Финальная feature matrix
├── notebooks/
│   ├── 01_eda.ipynb            # Разведочный анализ
│   ├── 02_features.ipynb       # Feature engineering эксперименты
│   ├── 03_model.ipynb          # Обучение модели
│   ├── 04_segmentation.ipynb   # Сегментация найденных
│   └── FINAL_pipeline.ipynb    # ⭐ Финальный воспроизводимый ноутбук
├── src/
│   ├── features/               # Модули feature engineering
│   ├── models/                 # Модули обучения
│   ├── evaluation/             # Метрики и графики
│   └── segmentation/           # Кластеризация
├── app/
│   └── streamlit_app.py        # Интерактивный дашборд
├── models/                     # Сохранённые веса (не в git)
├── reports/figures/            # Экспортированные графики для презентации
├── configs/                    # YAML конфиги
└── docs/                       # Документация проекта
    ├── CASE.md                 # Описание кейса
    ├── STACK.md                # Технический стек
    ├── SOLUTION.md             # Архитектура решения
    ├── FEATURES.md             # Описание всех фичей
    └── CRITERIA.md             # Критерии оценки
```

---

## Документация

| Файл | Содержание |
|------|-----------|
| [docs/CASE.md](docs/CASE.md) | Полное описание кейса и данных |
| [docs/SOLUTION.md](docs/SOLUTION.md) | Архитектура ML-решения |
| [docs/FEATURES.md](docs/FEATURES.md) | Все фичи с обоснованием |
| [docs/STACK.md](docs/STACK.md) | Технический стек |
| [docs/CRITERIA.md](docs/CRITERIA.md) | Критерии оценки жюри |

---

## Команда

| Имя | Роль |
|-----|------|
| TBD | TBD |
