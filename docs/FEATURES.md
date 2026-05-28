# Feature Engineering — полный список

Уровень агрегации: **карта (card_number)**  
Итого: ~80 фичей  
13M транзакций → ~105K строк (по одной на карту)

---

## Группа A: Объём и интенсивность

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `total_spend_kzt` | Суммарные траты за период | Бизнес тратит больше |
| `n_transactions` | Количество транзакций | Бизнес имеет больше операций |
| `n_active_days` | Дней с хотя бы 1 транзакцией | Бизнес активен регулярно |
| `tx_per_active_day` | Транзакций в день (в активные дни) | Интенсивность операционной деятельности |
| `avg_ticket_kzt` | Средний чек | Бизнес-закупки крупнее потребительских |
| `median_ticket_kzt` | Медианный чек | Устойчива к выбросам |
| `max_ticket_kzt` | Максимальный чек | Крупные оптовые закупки |
| `std_ticket_kzt` | Стандартное отклонение чека | Вариабельность трат |
| `p95_ticket_kzt` | 95-й перцентиль чека | Тяжёлый хвост распределения |
| `p95_p50_ratio` | p95 / p50 | Соотношение хвоста к медиане |

---

## Группа B: Diversity (разнообразие)

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `n_unique_merchants` | Количество уникальных мерчантов | ⚠️ EDA: business НИЖЕ (медиана 16 vs 37 у consumer). Бизнес работает с узким B2B-пулом (481/2165 мерчантов), consumer ходит везде. Фича работает в обратную сторону от исходной гипотезы. |
| `n_unique_mcc` | Количество уникальных MCC-категорий | ⚠️ EDA: business НИЖЕ (медиана 15 vs 32). Business специализирован на узком наборе B2B-MCC. Ящики почти не перекрываются — сильный сигнал. |
| `merchant_hhi` | Индекс Херфиндаля (концентрация трат по мерчантам) | EDA: business ВЫШЕ (медиана 0.224 vs 0.102 у consumer, q75 consumer < q25 business). Высокий = концентрация у B2B-поставщиков — верная направленность. |
| `top1_merchant_share` | Доля топ-1 мерчанта в тратах | Зависимость от одного поставщика |
| `top5_merchants_share` | Доля топ-5 мерчантов в тратах | Концентрация у ключевых поставщиков |
| `unique_merchants_per_month` | Avg уникальных мерчантов в месяц | Стабильность деловых связей |

---

## Группа C: MCC-based Business Signals ⭐ (ключевая группа)

### Классификация MCC

**B2B MCC — явный бизнес-сигнал:**
```
7372 — Software / SaaS (подписки на ПО)
5045 — Computers & IT equipment
7311 — Advertising services (Google Ads, Meta Ads)
2741 — Printing & publishing
5300 — Wholesale clubs / Оптовая торговля
5111 — Stationery / Office supplies
4214 — Freight / Logistics / Delivery
7392 — Consulting / Management services
5065 — Electrical equipment wholesale
5040 — Professional equipment
--- Добавлено по EDA (топ-20 MCC, business-dominant коды из §3.3 в 01_eda.ipynb): ---
5968 — Direct marketing / subscription services (Zoom, Shopify, Slack) ← +22.9% business delta
4816 — Computer network / hosting (Cloudflare, DigitalOcean, GoDaddy, Hetzner)
7399 — Business services NEC
8931 — Accounting / auditing / bookkeeping
7379 — Computer services / repair
5046 — Commercial equipment NEC
--- Эти 6 кодов дают delta business=22.9% vs consumer=1.1% → ratio 21× (лучше основного списка) ---
```

**Consumer MCC — потребительский сигнал:**
```
5812 — Restaurants / Eateries
5814 — Fast food
7832 — Movie theaters
7995 — Entertainment / Gambling
5411 — Grocery stores / Supermarkets
5311 — Department stores
```

**Mixed MCC — двойственные:**
```
5541 — Gas stations (АЗС)
4814 — Telecom / Mobile
4121 — Taxis / Ridesharing
5499 — Misc food stores
```

**Rental / Hospitality MCC — индикатор арендного бизнеса (добавлено после подсказки из чата хакатона):**
```
7011 — Hotels / Motels (перенесён из CONSUMER — на consumer-картах часто
       принадлежит тем кто сам сдаёт жильё и анализирует конкурентов)
7012 — Timeshares / Short-term apartment rentals (43K транзакций у consumer,
       0 мерчантов в reference — сильный сигнал скрытого Airbnb-бизнеса)
```

Также добавлено в B2B:
```
5099 — Durable goods wholesale NEC (54K у business, 115K у consumer —
       оптовые закупки, классический признак малого бизнеса)
```

### Фичи на основе MCC-классификации

| Фича | Описание |
|------|----------|
| `b2b_spend_share` | Доля трат на B2B-MCC |
| `b2b_tx_share` | Доля транзакций на B2B-MCC |
| `b2b_tx_count` | Абсолютное число B2B-транзакций |
| `n_unique_b2b_mcc` | Количество уникальных B2B-MCC |
| `consumer_spend_share` | Доля трат на Consumer-MCC |
| `consumer_tx_share` | Доля транзакций на Consumer-MCC |
| `mixed_spend_share` | Доля трат на Mixed-MCC |
| `b2b_recurring_share` | Доля B2B-трат, которые recurring (SaaS подписки) |
| `b2b_foreign_share` | Доля B2B-трат у иностранных мерчантов (Google Ads, AWS) |
| `rental_tx_share` | Доля транзакций на rental MCC (7011, 7012). ⭐ **NEW** — индикатор Airbnb-хостов среди consumer |
| `rental_spend_share` | Доля трат на rental MCC |
| `n_unique_rental_mcc` | Число уникальных rental MCC (0-2) |

---

## Группа D: Временные паттерны ⭐ (ключевая группа)

**Гипотеза:** Бизнес-расходы делаются в рабочее время. Потребительские — вечером и в выходные.

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `business_hours_share` | Доля транзакций пн-пт 9:00-18:00 | EDA: business 60.2% vs consumer 33.7% (1.8×) ✓ |
| `weekday_share` | Доля транзакций в будние дни | |
| `weekend_share` | Доля транзакций Сб-Вс | EDA: consumer 35.0% vs business 12.4% (2.8×). Высокое = consumer ✓ |
| `evening_share` | Доля 18:00-23:00 | Высокое = consumer |
| `night_share` | Доля 23:00-06:00 | ⚠️ EDA: business 14.6% vs consumer 3.2% (4.5×) — business НОЧЬЮ ВЫШЕ. Причина: SaaS-биллинг по UTC midnight = 2-5 утра Алматы. |
| `morning_share` | Доля 06:00-09:00 | Ранние закупки (рестораторы, рынок) |
| `hour_entropy` | Энтропия Шеннона по часам дня | Низкая = концентрация (бизнес), Высокая = размазана |
| `dow_entropy` | Энтропия по дням недели | |
| `business_hours_b2b_share` | B2B-транзакции в рабочие часы | Interaction-фича, сильный сигнал |
| `night_recurring_share` | Доля транзакций 00:00-06:00 с is_recurring=True | ⭐ **НОВАЯ по EDA**: business медиана 13.4% vs consumer медиана **0.0%**. Суперсигнал — большинство consumer-карт не имеют ночных recurring вообще. |
| `lunch_dip_ratio` | share@13h / mean(share@12h, share@14h) | **НОВАЯ по EDA**: business 0.727 vs consumer 1.0. У business обеденный провал (сотрудники не делают покупки в 13:00), у consumer нет. |
| `december_share` | Доля транзакций в декабре от общего объёма | **НОВАЯ по EDA**: consumer 19.8% vs business 15.7% (1.3×). Consumer — новогодний шоппинг, business — ровно. Слабый, но стабильный сигнал. |

---

## Группа E: Регулярность

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `recurring_share` | Доля is_recurring транзакций | Бизнес часто на SaaS/подписках |
| `recurring_amount_share` | Доля денег на recurring | |
| `n_unique_recurring_merchants` | Число уникальных recurring-мерчантов | |
| `inter_tx_time_median_days` | Медианный интервал между транзакциями | |
| `inter_tx_time_cv` | Coefficient of variation интервалов | Низкий = регулярный бизнес |
| `monthly_spend_cv` | Стабильность месячных трат | Низкий = стабильный оборот |

---

## Группа F: Канал и технологии

| Фича | Описание |
|------|----------|
| `online_share` | Доля online-транзакций |
| `offline_share` | Доля offline-транзакций |
| `tokenized_share` | Доля Apple Pay / Samsung Pay | ⚠️ EDA: business 58.5% vs consumer 38.5% — business ВЫШЕ. Вероятно, SaaS-мерчанты принимают оплату через Apple Pay. Интерпретация в FEATURES.md была ошибочной. |
| `online_b2b_share` | Online-транзакции × B2B-MCC |

---

## Группа G: География

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `n_unique_countries` | Число уникальных стран транзакций | |
| `foreign_tx_share` | Доля иностранных транзакций | |
| `foreign_b2b_share` | Доля иностранных B2B-транзакций | Google Ads, AWS, SaaS — иностранные |
| `kz_share` | Доля KZ-транзакций | |

---

## Группа H: Velocity / Burst

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `max_daily_tx_count` | Максимум транзакций за один день | День оптовой закупки |
| `max_daily_spend_kzt` | Максимальные траты за один день | |
| `n_days_5plus_tx` | Дней с 5+ транзакциями | |
| `n_days_10plus_tx` | Дней с 10+ транзакциями | |

---

## Группа I: Card-level статика

| Фича | Описание |
|------|----------|
| `card_tier` | Продуктовый уровень (категориальная) |
| `bank_name` | Банк-эмитент (категориальная) |
| `n_months_active` | Сколько месяцев из 6 активна карта |

---

## Группа J: Merchant Graph Features ⭐ (бонус/креатив)

Строим bipartite-граф: карты ↔ мерчанты.

| Фича | Описание | Как считать |
|------|----------|-------------|
| `business_merchant_overlap` | Доля мерчантов карты, которые также используют business-карты | Intersection / card_merchants |
| `consumer_merchant_overlap` | То же для типично consumer-мерчантов | |
| `merchant_signature_cosine` | Cosine similarity вектора "карта-мерчант" со средним business | TF-IDF по merchant_id |

**Зачем:** Если карта ходит к тем же поставщикам что и бизнес-карты — она с большей вероятностью бизнес. Это не уловимо стандартными фичами, но улавливается через граф.

---

## Приоритет фичей (обновлён по результатам Feature Engineering — 02_features.ipynb)

Медианы подтверждены на реальной feature matrix (105K карт). Корреляция с label по Pearson.

### Топ-фичи по mean |SHAP| (после фикса foreign_tx_share BUG)

| # | Фича | Business median | Consumer median | mean \|SHAP\| | Бизнес-смысл |
|---|------|-----------------|-----------------|--------------|--------------|
| 1 | `business_merchant_overlap` | 1.0 | 0.46 | **7.80** | ⭐ Карта пересекается с пулом мерчантов, которые посещают business-карты |
| 2 | `tokenized_share` | 58.5% | 38.5% | 1.32 | Apple Pay / Samsung Pay (SaaS-мерчанты принимают) |
| 3 | `evening_share` | 11.4% | 35.7% | 0.88 | Негативный: потребители активны 18-23ч |
| 4 | `recurring_amount_share` | 26.1% | 0.0% | 0.77 | SaaS-подписки крупными суммами |
| 5 | `n_unique_b2b_mcc` | высокий | низкий | 0.46 | Разнообразие B2B-категорий |
| 6 | `online_share` | высокий | низкий | 0.32 | Диджитал-бизнес |
| 7 | `merchant_signature_cosine` | высокий | низкий | 0.24 | Cosine similarity к среднему business-вектору |
| 8 | `weekend_share` | 12.4% | 35.0% | 0.14 | Негативный: бизнес не работает на выходных |
| 9 | `b2b_tx_count` | высокий | низкий | 0.09 | Абсолютное число B2B-транзакций |
| 10 | `weekday_share` | 87.6% | 65.0% | 0.08 | Бизнес работает пн-пт |
| 11 | `top5_merchants_share` | высокий | низкий | 0.08 | Концентрация у топ-5 поставщиков |
| 12 | `consumer_merchant_overlap` | 1.0 | 1.0 | 0.07 | ⚠️ Logic-bug: для consumer всегда = 1.0, для business вариабельно |
| 13 | `merchant_hhi` | 0.224 | 0.102 | 0.06 | 2.2× — концентрация по мерчантам |
| 14 | `b2b_recurring_share` | высокий | низкий | 0.05 | SaaS-подписки на B2B-мерчантах |
| 15 | `b2b_spend_share` | 83.6% | 0.0% | 0.04 | Доля трат на B2B-MCC |

### Текущая статистика топ-фичей по выборкам (после BUG fix)

| Фича | Business median | Consumer median | Top-50 (по score) |
|------|-----------------|-----------------|-------------------|
| `merchant_hhi` | 0.224 | 0.102 | **0.590** ⭐ |
| `b2b_spend_share` | 0.836 | 0.000 | **0.963** ⭐ |
| `foreign_tx_share` | **0.292** | **0.215** | **0.380** (после фикса BUG) |
| `b2b_foreign_share` | **0.406** | **0.000** | — (median 0.0 у consumer = сильный сигнал) |
| `recurring_amount_share` | 0.261 | 0.000 | 0.069 |
| `business_merchant_overlap` | 1.000 | 0.462 | 1.000 |
| `n_unique_merchants` | 16 | 37 | 6 |
| `weekday_share` | 0.876 | 0.650 | 0.896 |
| `evening_share` | 0.114 | 0.357 | 0.104 |

### Сломанные/слабые фичи (warning)

- ⚠️ `consumer_merchant_overlap` — для consumer-карт всегда = 1.0 по построению (logic bug в graph_features). Для модели бесполезна, но входит в feature matrix.
- ⚠️ `foreign_tx_share` ИСПРАВЛЕНА на 2026-05-28: был баг сравнения `country != "KZ"` (в данных = `"Kazakhstan"`). После фикса работает.
- ⚠️ `night_share`, `night_recurring_share` — timezone = Almaty (Q6). Не UTC-артефакт. Реальная ночная активность мерчантов. Возможно артефакт синтетического генератора — рекомендуется ablation test.

### Ключевые выводы

- **`business_merchant_overlap` — флагман модели** (mean |SHAP| = 7.8, в 5× выше всех остальных). Bipartite-граф фича из Group J оказалась мощнее всех остальных.
- **`tokenized_share`** работает как сигнал в **обратную** сторону от исходной гипотезы — бизнес выше потребителя из-за SaaS-мерчантов, принимающих Apple Pay.
- **`evening_share`** и **`weekend_share`** — главные негативные сигналы (потребительский паттерн).
- **`n_unique_merchants`** у business НИЖЕ — бизнес работает с узким пулом поставщиков.
