#!/usr/bin/env python3
"""Monitor dello stadio ``atomic_evidence_v3`` sul progetto LUNG.

Calcola progresso ed ETA della run in corso dalla tabella ``processing_runs``
e dai conteggi documenti per paziente.  Due stime:

1. **Throughput naive** — proporzionale al volume documentale residuo
   (indipendente dal numero di pazienti rimasti): ``doc_residui /
   (doc_fatti / finestra_temporale)``.
2. **Modello composizione** — regressione OLS sui pazienti completati:
   ``durata ≈ a·lab + b·nolab + c``, riadattata a ogni controllo.  Isola il
   costo dei documenti non-laboratorio (~4× quello lab) e l'overhead fisso per
   paziente.  Se la GPU è dedicata solo all'estrazione (niente contesa), la
   run tende al limite inferiore di questo modello.

Lettura read-only del registro: nessun impatto sulla run.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB = Path("/home/utente/Desktop/LUNG/emr_registry.db")
STAGE = "atomic_evidence_v3"
DONE_STATUS = ("completed", "completed_with_warnings")


def _fmt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.astimezone().strftime("%d/%m %H:%M")


def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help="Percorso del registro del progetto LUNG.",
    )
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        runs = con.execute(
            "SELECT patient_id, status, started_at, completed_at "
            "FROM processing_runs WHERE stage=?",
            (STAGE,),
        ).fetchall()
        doc_rows = con.execute(
            "SELECT patient_id, "
            "SUM(CASE WHEN document_type='laboratorio' THEN 1 ELSE 0 END) AS lab, "
            "SUM(CASE WHEN document_type!='laboratorio' THEN 1 ELSE 0 END) AS nolab "
            "FROM documents GROUP BY patient_id"
        ).fetchall()
        ev_rows = con.execute(
            "SELECT patient_id, COUNT(DISTINCT document_id) AS n "
            "FROM clinical_evidence GROUP BY patient_id"
        ).fetchall()
    finally:
        con.close()

    lab = {r["patient_id"]: r["lab"] for r in doc_rows}
    nolab = {r["patient_id"]: r["nolab"] for r in doc_rows}
    processed = {r["patient_id"]: r["n"] for r in ev_rows}

    done = [r for r in runs if r["status"] in DONE_STATUS]
    running = [r for r in runs if r["status"] == "running"]
    done_ids = {r["patient_id"] for r in done}
    run_ids = {r["patient_id"] for r in running}

    done_docs = sum(lab.get(r["patient_id"], 0) + nolab.get(r["patient_id"], 0)
                    for r in done)
    done_lab = sum(lab.get(r["patient_id"], 0) for r in done)
    done_nolab = done_docs - done_lab

    starts = [_iso(r["started_at"]) for r in done]
    ends = [_iso(r["completed_at"]) for r in done if r["completed_at"]]
    elapsed_min = (max(ends) - min(starts)).total_seconds() / 60.0

    # ---- documenti residui (in coda + quota residua del paziente in corso) --
    rem_lab = 0.0
    rem_nolab = 0.0
    n_pending = 0
    run_parts = []
    for r in running:
        pid = r["patient_id"]
        tot = lab.get(pid, 0) + nolab.get(pid, 0)
        prog = processed.get(pid, 0)
        frac = min(1.0, prog / tot) if tot else 1.0
        rem_lab += lab.get(pid, 0) * (1 - frac)
        rem_nolab += nolab.get(pid, 0) * (1 - frac)
        run_parts.append(f"{pid} {prog}/{tot}")
    for pid, l in lab.items():
        if pid in done_ids or pid in run_ids:
            continue
        n_pending += 1
        rem_lab += l
        rem_nolab += nolab.get(pid, 0)

    rem_docs = rem_lab + rem_nolab
    n_remaining = n_pending + len(running)

    # ---- stima 1: throughput naive ----------------------------------------
    throughput = done_docs / elapsed_min if elapsed_min > 0 else 0.0
    eta_naive = rem_docs / throughput if throughput > 0 else 0.0
    end_naive = now + timedelta(minutes=eta_naive)

    # ---- stima 2: modello composizione OLS (riadattato sui completati) -----
    if len(done) >= 6 and elapsed_min > 0:
        import numpy as np

        X = np.array([[lab.get(r["patient_id"], 0), nolab.get(r["patient_id"], 0)]
                      for r in done], dtype=float)
        y = np.array([(_iso(r["completed_at"]) - _iso(r["started_at"])).total_seconds() / 60.0
                      for r in done], dtype=float)
        A = np.column_stack([X, np.ones(len(done))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = coef
        pred = A @ coef
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

        eta_ols = a * rem_lab + b * rem_nolab + c * n_remaining
        eta_nolab = b * rem_nolab + c * n_remaining
        end_ols = now + timedelta(minutes=eta_ols)
        end_nolab = now + timedelta(minutes=eta_nolab)
        ols_ok = True
    else:
        a = b = c = r2 = 0.0
        eta_ols = eta_nolab = 0.0
        end_ols = end_nolab = None
        ols_ok = False

    print("─" * 66)
    print(f"LUNG · atomic_evidence_v3 · ora: {_fmt(now)}")
    print("─" * 66)
    print(f"pazienti   : {len(done)} fatti · {len(running)} in corso · "
          f"{n_pending} in coda")
    if run_parts:
        print(f"in corso   : {', '.join(run_parts)}")
    print(f"documenti  : {done_docs:,} fatti · {rem_docs:,.0f} residui "
          f"({rem_lab:,.0f} lab + {rem_nolab:,.0f} non-lab)")

    print(f"── throughput naive ──")
    print(f"finestra   : {elapsed_min:.0f} min per {done_docs:,} doc → "
          f"{throughput:.2f} doc/min")
    print(f"ETA        : {eta_naive / 60:.1f} h · fine {_fmt(end_naive)}")

    if ols_ok:
        print(f"── modello composizione (OLS su {len(done)} pazienti, R²={r2:.2f}) ──")
        print(f"costo      : {b * 60:.1f} s/non-lab · {a * 60:.1f} s/lab · "
              f"{c:.1f} min/paziente")
        print(f"compon.    : non-lab {b * rem_nolab:5.0f} min · lab "
              f"{a * rem_lab:5.0f} min · overhead {c * n_remaining:5.0f} min")
        print(f"ETA OLS    : {eta_ols / 60:.1f} h · fine {_fmt(end_ols)}")
        print(f"ETA nolab  : {eta_nolab / 60:.1f} h · fine {_fmt(end_nolab)} "
              f"(limite GPU dedicata)")
    else:
        print("── modello composizione ──")
        print("OLS non disponibile (pazienti completati insufficienti)")
    print("─" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
