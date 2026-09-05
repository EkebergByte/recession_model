import os
import sys
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    sys.exit(
        "BŁĄD: Brak FRED_API_KEY! Ustaw klucz w pliku .env lub GitHub Secrets."
    )

fred = Fred(api_key=API_KEY)


def fetch_point_in_time_series(code: str, is_revised: bool = False):
    if not is_revised:
        s = fred.get_series(code).dropna()
        return s.resample("MS").last().dropna()

    print(f"  • [ALFRED] Pobieranie historii rewizji dla: {code}...")
    try:
        df_rel = fred.get_series_all_releases(code)
        df_rel["date"] = pd.to_datetime(df_rel["date"])
        df_rel["realtime_start"] = pd.to_datetime(df_rel["realtime_start"])
        df_rel["value"] = pd.to_numeric(df_rel["value"], errors="coerce")

        first_release = df_rel.sort_values("realtime_start").drop_duplicates(
            subset=["date"], keep="first"
        )
        s = first_release.set_index("date")["value"].dropna()
        return s.resample("MS").last().dropna()
    except Exception as e:
        print(f"⚠️ Błąd ALFRED dla {code} ({e}). Pobieram standardowy szereg.")
        return fred.get_series(code).resample("MS").last().dropna()


def build_point_in_time_dataset():
    print("\n" + "=" * 80)
    print("🏛️ POBIERANIE DANYCH POINT-IN-TIME (ARCHIVAL FRED / ALFRED)")
    print("=" * 80)

    s_gs10 = fetch_point_in_time_series("GS10", is_revised=False)
    s_tb3m = fetch_point_in_time_series("TB3MS", is_revised=False)
    s_baa = fetch_point_in_time_series("BAA10Y", is_revised=False)
    s_claims = fetch_point_in_time_series("IC4WSA", is_revised=False)
    s_unrate = fetch_point_in_time_series("UNRATE", is_revised=True)
    s_permits = fetch_point_in_time_series("PERMIT", is_revised=True)
    s_usrec = fetch_point_in_time_series("USREC", is_revised=False)

    df = pd.DataFrame(index=s_gs10.index)

    # Rynki finansowe i wnioski o zasiłek (dostępne na koniec miesiąca T - Lag 0)
    df["yield_curve_10y3m"] = s_gs10 - s_tb3m
    df["yield_curve_delta6m"] = df["yield_curve_10y3m"] - df[
        "yield_curve_10y3m"
    ].shift(6)
    df["yield_curve_lag6m"] = df["yield_curve_10y3m"].shift(6)
    df["baa10y"] = s_baa
    df["baa_spread_lag6m"] = s_baa.shift(6)
    df["claims_yoy"] = s_claims.pct_change(12) * 100

    # Symulacja opóźnienia publikacji w historii (odpowiednik ffill na produkcji):
    # W miesiącu T rynek zna bezrobocie z T-1 oraz pozwolenia z T-2
    unrate_sahm = s_unrate.rolling(3).mean() - s_unrate.rolling(12).min()
    df["sahm_rule"] = unrate_sahm.shift(1)
    df["permits_yoy"] = (s_permits.pct_change(12) * 100).shift(2)

    df["usrec"] = s_usrec
    df = df.dropna()

    # NBER BLACKOUT WINDOW: Usunięcie ostatnich 18 miesięcy z próby treningowej
    NBER_BLACKOUT_MONTHS = 18
    df_clean = df.iloc[:-NBER_BLACKOUT_MONTHS].copy()

    print("\n✓ DANE PRZYGOTOWANE DO TRENINGU:")
    print(f"  • Początek próby               : {df_clean.index[0].strftime('%Y-%m')}")
    print(f"  • Koniec próby (NBER Cutoff)   : {df_clean.index[-1].strftime('%Y-%m')}")
    print(f"  • Wycięty bufor NBER (Blackout): Ostatnie {NBER_BLACKOUT_MONTHS} miesięcy")
    print(f"  • Efektywny rozmiar próby      : {len(df_clean)} miesięcy")
    return df_clean


def purged_walk_forward_validation(X, y, h, sample_weights):
    min_train = 240
    step = 24
    n_obs = len(X)

    y_true_all, y_pred_all = [], []

    for end_train in range(min_train, n_obs - step, step):
        start_test = end_train + h
        end_test = min(start_test + step, n_obs)
        if start_test >= n_obs:
            break

        X_tr, y_tr = X.iloc[:end_train], y.iloc[:end_train]
        w_tr = sample_weights[:end_train]
        X_te, y_te = X.iloc[start_test:end_test], y.iloc[start_test:end_test]

        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(C=1.0, random_state=42)
        clf.fit(X_tr_s, y_tr, sample_weight=w_tr)

        preds = clf.predict_proba(X_te_s)[:, 1]
        y_true_all.extend(y_te.values)
        y_pred_all.extend(preds)

    if len(y_true_all) > 0 and len(np.unique(y_true_all)) > 1:
        auc = roc_auc_score(y_true_all, y_pred_all)
        brier = brier_score_loss(y_true_all, y_pred_all)
        return auc, brier
    return np.nan, np.nan


def train_and_persist():
    df = build_point_in_time_dataset()

    configs = {
        3: ["sahm_rule", "claims_yoy", "baa10y", "yield_curve_10y3m"],
        6: [
            "yield_curve_10y3m",
            "yield_curve_delta6m",
            "baa10y",
            "claims_yoy",
        ],
        12: [
            "yield_curve_lag6m",
            "permits_yoy",
            "baa_spread_lag6m",
            "yield_curve_10y3m",
        ],
    }

    half_life_months = 240.0
    decay_rate = np.log(2) / half_life_months
    time_steps = np.arange(len(df))[::-1]
    weights = np.exp(-decay_rate * time_steps)
    weights = weights / np.mean(weights)

    bundle = {
        "models": {},
        "scalers": {},
        "configs": configs,
        "nber_cutoff": df.index[-1].strftime("%Y-%m"),
    }

    print("\n" + "=" * 80)
    print("🧠 TRENING LOGIT Z PURGED WALK-FORWARD VALIDATION (OOS)")
    print("=" * 80)

    for h, feats in configs.items():
        y = (
            df["usrec"]
            .rolling(window=h)
            .max()
            .shift(-h)
            .dropna()
            .astype(int)
        )
        X = df.loc[y.index, feats]
        w = weights[: len(X)]

        oos_auc, oos_brier = purged_walk_forward_validation(X, y, h, w)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(C=1.0, random_state=42)
        model.fit(X_scaled, y, sample_weight=w)

        bundle["models"][h] = model
        bundle["scalers"][h] = scaler

        print(f"► Horyzont {h:2}M:")
        print(f"   • Realny OOS ROC-AUC   : {oos_auc:.3f}")
        print(f"   • Realny OOS Brier Loss: {oos_brier:.3f}")
        print(
            f"   • Wagi beta             : {dict(zip(feats, np.round(model.coef_[0], 2)))}"
        )

    joblib.dump(bundle, "model_bundle.joblib")
    print("\n💾 [SUKCES] Zapisano model do 'model_bundle.joblib'.")
    print("=" * 80)


if __name__ == "__main__":
    train_and_persist()