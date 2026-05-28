"""
Phase 6a: Archetypes — 3 архетипа скрытых предпринимателей с реальными мерчантами.

Архетипы (выбраны после exploratory analysis topa-200):
  1. Wholesale Trader — крупные оптовые закупки (DurableGoods, логистика)
  2. Digital Marketer / SaaS user — Google Ads / Yandex Direct + Salesforce + Notion
  3. Rental / Airbnb host — наличие MCC 7011/7012
"""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    BUSINESS_CARDS_PATH, CONSUMER_CARDS_PATH, MERCHANTS_PATH,
    PROCESSED_DIR,
)

OUT_DIR = ROOT / "reports" / "diagnostics"


def _summarize_card(
    card_number: int,
    transactions: pl.DataFrame,
    merchants: pl.DataFrame,
    consumer_scored: pl.DataFrame,
) -> dict:
    card_tx = transactions.filter(pl.col("card_number") == card_number)
    if card_tx.is_empty():
        return None

    card_tx = card_tx.join(
        merchants.rename({"mcc": "_mcc_ref"}),
        on="merchant_id", how="left",
    )

    top_merch = (
        card_tx
        .group_by(["merchant_id", "merchant_name", "_mcc_ref", "merchant_country"])
        .agg([
            pl.len().alias("n_tx"),
            pl.col("transaction_amount_kzt").sum().alias("spend_kzt"),
            pl.col("is_recurring").sum().alias("n_recurring"),
        ])
        .sort("spend_kzt", descending=True)
        .head(6)
        .to_pandas()
    )

    scored_row = consumer_scored.filter(pl.col("card_number") == card_number).row(0, named=True)

    return {
        "card_number": card_number,
        "score": scored_row.get("business_score"),
        "combined_score": scored_row.get("combined_score"),
        "n_tx": len(card_tx),
        "total_spend_kzt": int(card_tx["transaction_amount_kzt"].sum()),
        "top_merchants": top_merch,
        "summary": {
            "b2b_spend_share": scored_row.get("b2b_spend_share"),
            "merchant_hhi": scored_row.get("merchant_hhi"),
            "foreign_tx_share": scored_row.get("foreign_tx_share"),
            "tokenized_share": scored_row.get("tokenized_share"),
            "rental_tx_share": scored_row.get("rental_tx_share"),
            "recurring_amount_share": scored_row.get("recurring_amount_share"),
            "n_unique_merchants": scored_row.get("n_unique_merchants"),
            "weekday_share": scored_row.get("weekday_share"),
            "evening_share": scored_row.get("evening_share"),
        },
    }


def render_archetype(label: str, hypothesis: str, product: str, summary: dict) -> str:
    s = summary["summary"]
    lines = [
        f"## {label}",
        "",
        f"**Hypothesis:** {hypothesis}",
        "",
        f"**Product to offer:** {product}",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Card number | `{summary['card_number']}` |",
        f"| PU-Bagging score | {summary['score']:.4f} |",
        f"| Combined (PU × Anomaly) score | {summary['combined_score']:.4f} |",
        f"| Total spend, 6 months | **{summary['total_spend_kzt']:,} ₸** |",
        f"| Transactions | {summary['n_tx']} |",
        f"| Unique merchants | {int(s['n_unique_merchants'])} (consumer median: 37) |",
        f"| B2B spend share | **{s['b2b_spend_share']*100:.1f}%** (consumer median: 0%) |",
        f"| Merchant HHI | {s['merchant_hhi']:.2f} (consumer median: 0.10) |",
        f"| Foreign tx share | {s['foreign_tx_share']*100:.1f}% (consumer: 22%) |",
        f"| Tokenized | {s['tokenized_share']*100:.1f}% |",
        f"| Recurring amount share | {s['recurring_amount_share']*100:.1f}% |",
        f"| Rental MCC share | {s['rental_tx_share']*100:.1f}% |",
        f"| Weekday share | {s['weekday_share']*100:.1f}% |",
        f"| Evening share | {s['evening_share']*100:.1f}% |",
        "",
        "**Top-6 merchants by spend:**",
        "",
        summary["top_merchants"].to_markdown(index=False, floatfmt=",.0f"),
    ]
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("PHASE 6a: ARCHETYPES")
    print("=" * 70)

    con_tx = pl.read_parquet(CONSUMER_CARDS_PATH)
    if "Is_recurring" in con_tx.columns:
        con_tx = con_tx.rename({"Is_recurring": "is_recurring"})
    if con_tx.schema.get("mcc") != pl.String:
        con_tx = con_tx.with_columns(pl.col("mcc").cast(pl.String))

    merchants = pl.read_parquet(MERCHANTS_PATH).with_columns(
        pl.col("mcc").cast(pl.String)
    )

    scored = pl.read_parquet(PROCESSED_DIR / "consumer_scored_v2.parquet")

    # ───────────────────────────────────────────────────────────────
    # Archetype 1: Wholesale Trader
    # — large bulk durable goods buys, narrow merchant base, low online tokenization
    # ───────────────────────────────────────────────────────────────
    wholesale = (
        scored
        .sort("combined_score", descending=True)
        .filter(
            (pl.col("merchant_hhi") > 0.7)
            & (pl.col("n_unique_merchants") <= 5)
            & (pl.col("tokenized_share") < 0.6)
            & (pl.col("recurring_amount_share") < 0.2)
        )
        .head(1)
    )

    # ───────────────────────────────────────────────────────────────
    # Archetype 2: Digital Marketer / SaaS-heavy
    # — high recurring (SaaS subscriptions), high foreign (Google Ads / Notion / Salesforce)
    # ───────────────────────────────────────────────────────────────
    digital = (
        scored
        .sort("combined_score", descending=True)
        .filter(
            (pl.col("recurring_amount_share") > 0.3)
            & (pl.col("foreign_tx_share") > 0.4)
            & (pl.col("rental_tx_share") < 0.01)
        )
        .head(1)
    )

    # ───────────────────────────────────────────────────────────────
    # Archetype 3: Rental / Airbnb host
    # ───────────────────────────────────────────────────────────────
    rental = (
        scored
        .sort("combined_score", descending=True)
        .filter(pl.col("rental_tx_share") > 0.03)
        .head(1)
    )
    if rental.is_empty():
        rental = scored.sort("rental_tx_share", descending=True).head(1)

    # ─── Output ────────────────────────────────────────────────────
    archetypes_md = ["# Hidden Entrepreneur Archetypes", "",
                     "Three case studies from cards ranked highest by combined score.", ""]

    archetype_data = []
    for label, hyp, product, df in [
        ("Archetype 1 — Wholesale Trader",
         "Cardholder makes a small number of very large bulk purchases at wholesale "
         "merchants (durable goods, logistics). Likely small importer / reseller "
         "running a trading business with extremely narrow supplier base.",
         "Merchant acquiring + working capital loan + international transfers",
         wholesale),
        ("Archetype 2 — Digital Marketer / SaaS Operator",
         "Cardholder runs a small marketing/IT operation: recurring payments to "
         "advertising platforms (Google Ads / Yandex Direct), SaaS tools "
         "(Salesforce, Notion, Slack), hosted abroad. Likely solo consultant or "
         "small agency.",
         "Multi-currency business card + cashback on ad spend + corporate SaaS rates",
         digital),
        ("Archetype 3 — Rental / Airbnb Host",
         "Cardholder spends on short-term rental MCCs (hotels/timeshares) — "
         "indicates either active analysis of competitors or actual purchase of "
         "rental nights to resell. Combined with hospitality services.",
         "Property-management card + dynamic pricing tools + insurance bundle",
         rental),
    ]:
        if df.is_empty():
            print(f"[{label}] no card matched; skipping")
            continue
        card_num = df["card_number"][0]
        summary = _summarize_card(card_num, con_tx, merchants, scored)
        if summary is None:
            continue
        md = render_archetype(label, hyp, product, summary)
        archetypes_md.append(md)
        archetypes_md.append("\n---\n")
        archetype_data.append({
            "label": label, "hypothesis": hyp, "product": product,
            "card_number": int(summary["card_number"]),
            "score": float(summary["score"]) if summary["score"] is not None else None,
            "combined_score": float(summary["combined_score"]),
            "n_tx": int(summary["n_tx"]),
            "total_spend_kzt": int(summary["total_spend_kzt"]),
            "summary": {k: (float(v) if v is not None else None) for k, v in summary["summary"].items()},
            "top_merchants": summary["top_merchants"].to_dict(orient="records"),
        })
        print(f"\n{'─' * 70}\n{md}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "archetypes.md").write_text("\n".join(archetypes_md), encoding="utf-8")

    import json
    with open(OUT_DIR / "archetypes.json", "w", encoding="utf-8") as f:
        json.dump(archetype_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved: {OUT_DIR / 'archetypes.md'}")
    print(f"Saved: {OUT_DIR / 'archetypes.json'}")


if __name__ == "__main__":
    main()
