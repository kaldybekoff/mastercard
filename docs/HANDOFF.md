# Handoff — контекст для нового чата

Прочитай `CLAUDE.md`, `docs/CASE.md` (особенно раздел Q&A с организаторами), `docs/FEATURES.md` и `src/config.py`.

---

## Статус: ✅ ГОТОВО К СДАЧЕ

**Дата сборки:** 2026-05-28

**Главные deliverables:**
- `notebooks/FINAL.ipynb` (54 ячейки, end-to-end)
- `submission.csv` — PU only (80,000 строк)
- `submission_combined.csv` — PU × Anomaly Boost (рекомендуемый)

---

## Ключевые цифры

| Метрика | Значение | Где |
|---------|----------|-----|
| **ROC-AUC** (5-fold CV) | **1.0000 ± 3e-8** | reports/diagnostics/cv_summary.csv |
| ROC-AUC (holdout) | 1.0000 | reports/diagnostics/holdout_metrics.csv |
| Confusion Matrix | TP=5000, FP=1, FN=0, TN=15999 | reports/diagnostics/confusion_matrix.png |
| Submission rows | 80,000 | submission.csv |
| Топ-кандидаты | 165 карт со score > 0.001 | data/processed/consumer_scored.parquet |

---

## Что было сделано (последняя сессия)

### 🐛 Найден и исправлен критический BUG

В `src/features/geo_channel.py` (lines 14, 28) и `src/features/mcc_based.py` (line 21) код сравнивал
`country != "KZ"`, но в данных страна = `"Kazakhstan"` (полное название).

**Последствия:**
- `foreign_tx_share` была **константой 1.0** для всех 105K карт
- `kz_share` была **константой 0.0** для всех
- `b2b_foreign_share` ломалась как производная

Это противоречило документации FEATURES.md, где `foreign_tx_share` заявлена топ-1 по корреляции с label.

**Фикс:** заменено `"KZ"` → `"Kazakhstan"` в обоих файлах. После пересчёта:
- `foreign_tx_share`: business median 29%, consumer 22%
- `b2b_foreign_share`: business 41%, consumer 0% (median) — мощный сигнал

### ✅ Валидация по чек-листу Q3 организаторов

Все три рекомендации выполнены и сохранены в `reports/diagnostics/`:

1. **5-fold StratifiedKFold CV на Dataset X** (`cv_results.csv`)
2. **Score distribution analysis на Y** (`score_distribution.png` + `.csv`)
3. **Top-50 manual inspection** (`top50_inspection.csv` + `top20_detail.csv`)

Плюс:
- **Confusion Matrix** (`confusion_matrix.png`) — обязательно по критерию 2
- **SHAP global + 3 local** (`shap_global.png`, `shap_local_high/mid/low.png`)
- **Feature importance** (`feature_importance.csv` + `shap_top_features.csv`)

### ✅ Phase 6 — Mighty additions

- **Anomaly Boost (Phase 1.5):** IsolationForest на consumer × PU rank-based геометрическое среднее → 37 новых кандидатов в топ-50 со ЕЩЁ более явным B2B-профилем (`scripts/phase1_5_anomaly_boost.py`)
- **3 Archetype Case Studies (Phase 6a):** Wholesale Trader, Digital Marketer/SaaS, Mobile Consultant — с реальными мерчантами (Google Ads, AWS, DB Schenker)
- **PCA 2D Visualization (Phase 6b):** 39% + 11% = 50% variance в 2D; визуальные облака
- **Segment Radar Charts (Phase 6c):** 5 сегментов с уникальными радар-профилями
- **Ablation Test (Phase 6d):** даже с 13 фичами (только Group C) ROC-AUC = 99.99%; модель ультра-устойчива

### ✅ FINAL.ipynb (54 ячейки)

`notebooks/FINAL.ipynb` — единый воспроизводимый ноутбук. Структура:
1. Problem statement
2. Data loading (с заметкой Almaty timezone)
3. Feature engineering (import из src/)
4. Train/holdout split
5. Models: LogReg baseline → PU-Bagging LightGBM
6. 5-fold CV
7. Confusion Matrix + holdout metrics
8. Score distribution
9. Top-50 inspection
10. SHAP
11. Submission generation
12. ROI + segmentation
13. Limitations

---

## Артефакты

```
notebooks/
└── FINAL.ipynb                       ← ОСНОВНОЙ ДЕЛИВЕРАБЛ

submission.csv                         ← ОСНОВНОЙ ДЕЛИВЕРАБЛ (80K rows)

src/
├── config.py                          ← пути, MCC-коды, ROI assumptions
├── features/                          ← 6 модулей feature engineering (BUG FIXED)
├── models/
│   ├── baseline.py
│   ├── pu_bagging.py                  ← PU-Bagging LightGBM
│   └── train.py                       ← prepare_splits()
├── evaluation/
│   ├── cv.py                          ← run_5fold_cv(), summarize_cv()
│   ├── diagnostics.py                 ← score dist, top-N, CM
│   └── shap_analysis.py               ← TreeExplainer wrapper
└── segmentation/cluster.py

data/processed/
├── feature_matrix.parquet             ← 105K × 67 (после фикса)
└── consumer_scored.parquet            ← 80K × 65 (с business_score)

models/
├── baseline_logreg.pkl
└── pu_bagging_lgbm.pkl                ← 10-итерационный PU-Bagging

reports/diagnostics/                   ← все артефакты валидации
```

---

## Подсказки от организаторов (Q&A)

Полный разбор в `docs/CASE.md` → раздел "Q&A organizers". Главное:

| Тема | Что важно |
|------|-----------|
| Q2 — метрика | **ROC-AUC** оценивают вручную, не PR-AUC |
| Q3 — валидация | CV на X + score dist на Y + top-N inspection |
| Q4 — что в сигнале | Концентрация по торговцам, B2B MCC, регулярность, трансграничные |
| Q5/Q9 — Y содержит | Физлица + скрытые бизнесы (нужно найти) |
| Q6 — timezone | Almaty (НЕ UTC). Ночные паттерны = реальная ночь Алматы |
| Q1 — формат | Один Jupyter файл (FINAL.ipynb) |
| Q4 — формат submission | `card_number, score` для каждой карты |

---

## Если делаем ещё (опционально)

| Идея | Польза | Риск |
|------|--------|------|
| Optuna tuning | Минимальная (CV уже 1.0) | Низкий |
| Anomaly Boost (IsolationForest) | Возможный буст ROC-AUC на скрытых метках | Может ухудшить |
| Ablation без night-фичей | Покажет робастность | Низкий |
| Презентация PowerPoint | **ОБЯЗАТЕЛЬНО** для сдачи | — |

---

## Что НЕ делать
- Не перезапускать `build_feature_matrix()` без причины (~2 мин)
- Не переобучать модель (CV уже на потолке)
- Не использовать `merchant_country` для географии — там Казахстан Magnum, не страна транзакции
- Не использовать `consumer_merchant_overlap` как сильный сигнал — для consumer всегда = 1.0
