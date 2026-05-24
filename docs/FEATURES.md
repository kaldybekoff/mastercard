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
| `n_unique_merchants` | Количество уникальных мерчантов | Бизнес работает со множеством поставщиков |
| `n_unique_mcc` | Количество уникальных MCC-категорий | Диверсификация операций |
| `merchant_hhi` | Индекс Херфиндаля (концентрация трат по мерчантам) | Низкий = много поставщиков, Высокий = концентрация |
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

---

## Группа D: Временные паттерны ⭐ (ключевая группа)

**Гипотеза:** Бизнес-расходы делаются в рабочее время. Потребительские — вечером и в выходные.

| Фича | Описание | Бизнес-логика |
|------|----------|---------------|
| `business_hours_share` | Доля транзакций пн-пт 9:00-18:00 | Рабочие закупки в рабочее время |
| `weekday_share` | Доля транзакций в будние дни | |
| `weekend_share` | Доля транзакций Сб-Вс | Высокое = consumer |
| `evening_share` | Доля 18:00-23:00 | Высокое = consumer |
| `night_share` | Доля 23:00-06:00 | |
| `morning_share` | Доля 06:00-09:00 | Ранние закупки (рестораторы, рынок) |
| `hour_entropy` | Энтропия Шеннона по часам дня | Низкая = концентрация (бизнес), Высокая = размазана |
| `dow_entropy` | Энтропия по дням недели | |
| `business_hours_b2b_share` | B2B-транзакции в рабочие часы | Interaction-фича, сильный сигнал |

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
| `tokenized_share` | Доля Apple Pay / Samsung Pay |
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

## Приоритет фичей (по ожидаемой важности)

1. `b2b_spend_share` — сильнейший одиночный сигнал
2. `business_hours_share` — время работы
3. `hour_entropy` / `dow_entropy` — регулярность паттерна
4. `n_unique_merchants` — breadth деловых связей
5. `b2b_recurring_share` — SaaS/подписки = бизнес
6. `foreign_b2b_share` — Google Ads, AWS
7. `merchant_hhi` — структура расходов
8. `business_hours_b2b_share` — interaction фича
9. `monthly_spend_cv` — стабильность оборота
10. `business_merchant_overlap` — граф-фича

> После обучения модели — верификация через SHAP. Если топ-10 SHAP фичей совпадает с этим списком → модель учит правильный сигнал.
