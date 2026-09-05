import html
import json
import os
import sys
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    sys.exit("BŁĄD: Brak FRED_API_KEY w pliku .env!")

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

fred = Fred(api_key=API_KEY)
STATE_FILE = "fred_state.json"
BUNDLE_FILE = "model_bundle.joblib"

SERIES = {
    "gs10": "GS10",
    "tb3m": "TB3MS",
    "baa10y": "BAA10Y",
    "unrate": "UNRATE",
    "claims_4w": "IC4WSA",
    "permits": "PERMIT",
}


def send_alert(message_html: str):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try:
            r = requests.post(
                url,
                json={
                    "chat_id": TG_CHAT_ID,
                    "text": message_html,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"Błąd wysyłki Telegram: {e}")
    print(f"\n[RAPORT SYSTEMOWY]:\n{message_html}")


def check_and_run():
    if not os.path.exists(BUNDLE_FILE):
        sys.exit(f"Brak pliku '{BUNDLE_FILE}'! Uruchom: python train_models.py")

    last_state = (
        json.load(open(STATE_FILE, encoding="utf-8"))
        if os.path.exists(STATE_FILE)
        else {}
    )
    current_state = {}
    changed_series = []

    # 1. Detekcja zmian metadanych w FRED
    for name, s_id in SERIES.items():
        info = fred.get_series_info(s_id)
        last_up = info.get("last_updated")
        current_state[s_id] = last_up
        if last_state.get(s_id) != last_up:
            changed_series.append((name, s_id, info.get("title")))

    if not changed_series and "last_probabilities" in last_state:
        print("☕ [FRED] Brak nowych danych od ostatniego sprawdzenia.")
        return

    updated_names = (
        ", ".join([item[0] for item in changed_series])
        if changed_series
        else "Inicjalizacja systemu"
    )
    print(f"⚡ Wykryto nowe dane: {updated_names}")

    # 2. Pobranie danych
    raw = {k: fred.get_series(v).dropna() for k, v in SERIES.items()}

    s_gs10 = raw["gs10"].resample("MS").last().dropna()
    s_tb3m = raw["tb3m"].resample("MS").last().dropna()
    s_baa = raw["baa10y"].resample("MS").last().dropna()
    s_unrate = raw["unrate"].resample("MS").last().dropna()
    s_claims = raw["claims_4w"].resample("MS").last().dropna()
    s_permits = raw["permits"].resample("MS").last().dropna()

    # 3. Wyliczanie cech na surowych seriach (BEZ PODWÓJNEGO SHIFT)
    # Rynki finansowe i zasiłki (stan bieżący T)
    yc_10y3m = s_gs10 - s_tb3m
    yc_delta6m = yc_10y3m - yc_10y3m.shift(6)
    yc_lag6m = yc_10y3m.shift(6)
    baa_lag6m = s_baa.shift(6)
    claims_yoy = s_claims.pct_change(12) * 100

    # Makroekonomia: wyliczamy wskaźniki na ich własnych dostępnych datach
    # s_unrate kończy się na T-1, s_permits na T-2. ffill() przeniesie je do T.
    sahm_rule = s_unrate.rolling(3).mean() - s_unrate.rolling(12).min()
    permits_yoy = s_permits.pct_change(12) * 100

    features_df = pd.DataFrame(
        {
            "yield_curve_10y3m": yc_10y3m,
            "yield_curve_delta6m": yc_delta6m,
            "yield_curve_lag6m": yc_lag6m,
            "baa10y": s_baa,
            "baa_spread_lag6m": baa_lag6m,
            "permits_yoy": permits_yoy,
            "claims_yoy": claims_yoy,
            "sahm_rule": sahm_rule,
        }
    ).ffill()

    latest_row = features_df.iloc[[-1]]
    latest_date = features_df.index[-1].strftime("%Y-%m-%d")

    # 4. Predykcja
    bundle = joblib.load(BUNDLE_FILE)
    new_probs = {}
    for h, feats in bundle["configs"].items():
        vec = latest_row[feats]
        vec_scaled = bundle["scalers"][h].transform(vec)
        prob = bundle["models"][h].predict_proba(vec_scaled)[0][1] * 100
        new_probs[str(h)] = round(prob, 2)

    old_probs = last_state.get("last_probabilities", {})

    # 5. Raport HTML
    msg = f"📊 <b>RAPORT MAKROEKONOMICZNY USA (POINT-IN-TIME)</b>\n"
    msg += f"Stan danych na: <code>{latest_date}</code>\n"
    msg += f"Zaktualizowano: <code>{html.escape(updated_names)}</code>\n\n"
    msg += f"🏛️ <b>Szacowane Ryzyko Recesji (Nowcast):</b>\n"

    for h in [3, 6, 12]:
        p_new = new_probs[str(h)]
        p_old = old_probs.get(str(h), p_new)
        delta = p_new - p_old
        sign = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        delta_str = f" (Zmiana: {sign})" if p_old != p_new else ""

        if p_new < 15.0:
            ico = "🟢"
        elif p_new < 35.0:
            ico = "🟡"
        elif p_new < 55.0:
            ico = "🟠"
        else:
            ico = "🔴"

        msg += f"{ico} <b>Horyzont {h}M:</b> <code>{p_new:.2f}%</code>{delta_str}\n"

    yc = latest_row["yield_curve_10y3m"].values[0]
    sahm = latest_row["sahm_rule"].values[0]
    claims = latest_row["claims_yoy"].values[0]
    baa = latest_row["baa10y"].values[0]

    msg += (
        f"\n🔍 <b>Kluczowe Odczyty (Ostatnie Dostępne):</b>\n"
        f" • Sahm Rule: <code>{sahm:.2f} pkt</code>\n"
        f" • Spread 10Y-3M: <code>{yc:.2f}%</code>\n"
        f" • Zasiłki YoY: <code>{claims:.1f}%</code>\n"
        f" • Baa Spread: <code>{baa:.2f}%</code>"
    )

    send_alert(msg)

    # 6. Zapis stanu
    current_state["last_probabilities"] = new_probs
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_state, f, indent=2)


if __name__ == "__main__":
    check_and_run()