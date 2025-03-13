import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, percentileofscore
import math

# -------------------------------
# Thalex API details
# -------------------------------
BASE_URL = "https://thalex.com/api/v2/public"
instruments_endpoint = "instruments"  # Endpoint for fetching available instruments
url_instruments = f"{BASE_URL}/{instruments_endpoint}"
mark_price_endpoint = "mark_price_historical_data"
url_mark_price = f"{BASE_URL}/{mark_price_endpoint}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"

# Dictionary for any rolling window configuration (currently one 7D window)
windows = {"7D": "vrp_7d"}

# For backward compatibility, default expiration date if not selected
DEFAULT_EXPIRY_STR = "28MAR25"

# Calculate actual time to expiry (using default if needed)
expiry_date = dt.datetime.strptime(DEFAULT_EXPIRY_STR, "%d%b%y")
current_date = dt.datetime.now()
days_to_expiry = (expiry_date - current_date).days
T_YEARS = days_to_expiry / 365  # Convert days to years

def params(instrument_name):
    now = dt.datetime.now()
    start_dt = now - dt.timedelta(days=7)
    return {
        "from": int(start_dt.timestamp()),
        "to": int(now.timestamp()),
        "resolution": "5m",
        "instrument_name": instrument_name,
    }

# Column names returned by the Thalex API for mark price data
COLUMNS = [
    "ts",
    "mark_price_open",
    "mark_price_high",
    "mark_price_low",
    "mark_price_close",
    "iv_open",
    "iv_high",
    "iv_low",
    "iv_close",
]

###########################################
# EXPIRATION DATE HELPER FUNCTIONS (UPDATED)
###########################################
def get_valid_expiration_options(current_date=None):
    """Return the valid expiration day options based on today's date."""
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < 14:
        return [14, 28]
    elif current_date.day < 28:
        return [28]
    else:
        return [14, 28]

def compute_expiry_date(selected_day, current_date=None):
    """
    Compute the expiration date based on the selected day.
    - If today is before the chosen day, use the current month.
    - Otherwise, roll over to the next month.
    """
    if current_date is None:
        current_date = dt.datetime.now()
    if current_date.day < selected_day:
        try:
            expiry = current_date.replace(day=selected_day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            st.error("Invalid expiration date for current month.")
            return None
    else:
        year = current_date.year + (current_date.month // 12)
        month = (current_date.month % 12) + 1
        try:
            expiry = dt.datetime(year, month, selected_day)
        except ValueError:
            st.error("Invalid expiration date for next month.")
            return None
    return expiry

###########################################
# CREDENTIALS & LOGIN FUNCTIONS (from text files)
###########################################
def load_credentials():
    try:
        with open("usernames.txt", "r") as f_user:
            usernames = [line.strip() for line in f_user if line.strip()]
        with open("passwords.txt", "r") as f_pass:
            passwords = [line.strip() for line in f_pass if line.strip()]
        if len(usernames) != len(passwords):
            st.error("The number of usernames and passwords do not match.")
            return {}
        creds = dict(zip(usernames, passwords))
        return creds
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.title("Please Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            creds = load_credentials()
            if username in creds and creds[username] == password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
        st.stop()

###########################################
# INSTRUMENTS FETCHING & FILTERING FUNCTIONS
###########################################
def fetch_instruments():
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type, expiry_str):
    filtered = [
        inst["instrument_name"] for inst in instruments
        if inst["instrument_name"].startswith(f"BTC-{expiry_str}") and inst["instrument_name"].endswith(f"-{option_type}")
    ]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    response = requests.get(url_mark_price, params=params(instrument_name))
    if response.status_code != 200:
        return None
    data = response.json()
    marks = get_in(["result", "mark"])(data)
    if not marks:
        return None
    df = pd.DataFrame(marks, columns=COLUMNS)
    df = df.sort_values("ts")
    return df["iv_close"].iloc[-1]

def get_filtered_instruments(spot_price, expiry_str=DEFAULT_EXPIRY_STR, t_years=T_YEARS, multiplier=1):
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C", expiry_str)
    puts_all = get_option_instruments(instruments_list, "P", expiry_str)
    if not calls_all:
        raise Exception(f"No call instruments found for expiry {expiry_str}.")
    strike_list = []
    for inst in calls_all:
        parts = inst.split("-")
        if len(parts) >= 3 and parts[2].isdigit():
            strike_list.append((inst, int(parts[2])))
    if not strike_list:
        raise Exception(f"No valid call instruments with strikes found for expiry {expiry_str}.")
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    if not strikes:
        raise Exception("No strikes available for filtering.")
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument.")
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    filtered_calls = [inst for inst in calls_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_calls.sort(key=lambda x: int(x.split("-")[2]))
    filtered_puts.sort(key=lambda x: int(x.split("-")[2]))
    if not filtered_calls and not filtered_puts:
        raise Exception("No instruments left after applying the theoretical range filter.")
    return filtered_calls, filtered_puts

###########################################
# DATA FETCHING FUNCTIONS
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
    instruments = list(instruments_tuple)
    df = (
        pipe(
            {name: requests.get(url_mark_price, params=params(name)) for name in instruments},
            valmap(requests.Response.json),
            valmap(get_in(["result", "mark"])),
            valmap(curry(pd.DataFrame, columns=COLUMNS)),
            valfilter(lambda df: not df.empty),
            pd.concat,
        )
        .droplevel(1)
        .reset_index(names=["instrument_name"])
        .assign(date_time=lambda df: pd.to_datetime(df["ts"], unit="s")
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York"))
        .assign(k=lambda df: df["instrument_name"].map(lambda s: int(s.split("-")[2]) if len(s.split("-")) >= 3 and s.split("-")[2].isdigit() else np.nan))
        .assign(option_type=lambda df: df["instrument_name"].str.split("-").str[-1])
    )
    return df

@st.cache_data(ttl=30)
def fetch_ticker(instrument_name):
    params = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

###########################################
# UPDATED KRAKEN DATA FETCH (Dual Timeframe)
###########################################
def fetch_kraken_data():
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=365)
    
    # Fetch 5m data for real-time updates (last 30 days)
    ohlcv_5m = kraken.fetch_ohlcv("BTC/USD", timeframe="5m",
                                   since=int(start_dt.timestamp()) * 1000,
                                   limit=3000)
    
    # Fetch daily data for volatility calculations
    ohlcv_1d = kraken.fetch_ohlcv("BTC/USD", timeframe="1d",
                                   since=int(start_dt.timestamp()) * 1000)
    
    # Process 5m data
    df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_5m["date_time"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
    df_5m = df_5m.set_index("date_time")
    
    # Process daily data
    df_1d = pd.DataFrame(ohlcv_1d, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df_1d["date_time"] = pd.to_datetime(df_1d["timestamp"], unit="ms")
    df_1d = df_1d.set_index("date_time")
    
    # Combine datasets and remove duplicate timestamps
    full_df = pd.concat([df_5m, df_1d], axis=0).sort_index()
    return full_df[~full_df.index.duplicated()]

###########################################
# PARKINSON VOLATILITY CALCULATION
###########################################
def calculate_parkinson_volatility(price_data, window_days=7, annualize_days=365):
    """
    Calculate volatility using Parkinson's High-Low Range estimator.
    """
    if price_data.empty or not {'high', 'low'}.issubset(price_data.columns):
        return np.nan

    # Use "date_time" column if available, otherwise assume the index is datetime
    if "date_time" in price_data.columns:
        daily_data = price_data.resample('D', on='date_time').agg({
            'high': 'max',
            'low': 'min'
        }).dropna()
    else:
        daily_data = price_data.resample('D').agg({
            'high': 'max',
            'low': 'min'
        }).dropna()

    if len(daily_data) < window_days:
        return np.nan

    # Calculate daily log high-low ratio
    daily_data['log_hl_ratio'] = np.log(daily_data['high'] / daily_data['low'])

    # Calculate Parkinson variance components
    parkinson_var = daily_data['log_hl_ratio'].rolling(
        window=window_days, min_periods=1
    ).apply(lambda x: np.sum(x**2) / (4 * np.log(2) * window_days), raw=True)

    # Annualize and convert to volatility
    annualized_vol = np.sqrt(parkinson_var * annualize_days)

    # Reindex to match the original timestamps; if "date_time" is not in columns, align with the index
    if "date_time" in price_data.columns:
        return annualized_vol.reindex(price_data.index, method='ffill')
    else:
        return annualized_vol.reindex(price_data.index, method='ffill')
    
def compute_daily_realized_volatility(df):
    """Calculate daily Parkinson volatility values."""
    # Check if 'date_time' exists; if not, use the index for resampling
    if 'date_time' in df.columns:
        df_daily = df.resample('D', on='date_time').agg({
            'high': 'max',
            'low': 'min'
        }).dropna()
    else:
        df_daily = df.resample('D').agg({
            'high': 'max',
            'low': 'min'
        }).dropna()

    daily_vols = []
    for i in range(1, len(df_daily) + 1):
        window = df_daily.iloc[max(0, i - 7):i]
        if len(window) < 1:
            continue
        vol = calculate_parkinson_volatility(window, window_days=len(window))
        daily_vols.append(vol.iloc[-1] if not vol.empty else np.nan)
    
    return [v for v in daily_vols if not np.isnan(v)]


def compute_daily_average_iv(df_iv_agg):
    daily_iv = df_iv_agg["iv_mean"].resample("D").mean(numeric_only=True).dropna().tolist()
    return daily_iv

def compute_historical_vrp(daily_iv, daily_rv):
    """Compute historical VRP using Parkinson volatility (variance difference)."""
    n = min(len(daily_iv), len(daily_rv))
    return [(iv ** 2) - (rv ** 2) for iv, rv in zip(daily_iv[:n], daily_rv[:n])]

###########################################
# OPTION DELTA CALCULATION FUNCTION
###########################################
def compute_delta(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        T = 0.0001
    K = row["k"]
    sigma = row["iv_close"]
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    return norm.cdf(d1) if row["option_type"] == "C" else norm.cdf(d1) - 1

###########################################
# DELTA-BASED DYNAMIC REGIME FUNCTIONS
###########################################
def compute_average_delta(df_calls, df_puts, S):
    if "delta" not in df_calls.columns:
        df_calls["delta"] = df_calls.apply(lambda row: compute_delta(row, S), axis=1)
    if "delta" not in df_puts.columns:
        df_puts["delta"] = df_puts.apply(lambda row: compute_delta(row, S), axis=1)
    df_calls_mean = (
        df_calls.groupby("date_time", as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "call_delta_avg"})
    )
    df_puts_mean = (
        df_puts.groupby("date_time", as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "put_delta_avg"})
    )
    df_merged = pd.merge(df_calls_mean, df_puts_mean, on="date_time", how="outer").sort_values("date_time")
    df_merged["delta_diff"] = df_merged["call_delta_avg"] - df_merged["put_delta_avg"]
    return df_merged

def rolling_percentile_zones(df, column="delta_diff", window="1D", lower_percentile=33, upper_percentile=66):
    df = df.set_index("date_time").sort_index()
    def percentile_in_window(x, q):
        return np.percentile(x, q)
    df["rolling_lower_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, lower_percentile), raw=False)
    df["rolling_upper_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, upper_percentile), raw=False)
    return df.reset_index()

def classify_regime(row):
    if pd.isna(row["rolling_lower_zone"]) or pd.isna(row["rolling_upper_zone"]):
        return "Neutral"
    if row["delta_diff"] > row["rolling_upper_zone"]:
        return "Risk-On"
    elif row["delta_diff"] < row["rolling_lower_zone"]:
        return "Risk-Off"
    else:
        return "Neutral"

def compute_gamma(row, S):
    try:
        expiry_str = row["instrument_name"].split("-")[1]
        expiry_date = dt.datetime.strptime(expiry_str, "%d%b%y")
        expiry_date = expiry_date.replace(tzinfo=row["date_time"].tzinfo)
    except Exception:
        return np.nan
    T = (expiry_date - row["date_time"]).total_seconds() / (365 * 24 * 3600)
    if T <= 0:
        return np.nan
    K = row["k"]
    sigma = row["iv_close"]
    if sigma <= 0:
        return np.nan
    try:
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    except Exception:
        return np.nan
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def compute_gex(row, S, oi):
    gamma = compute_gamma(row, S)
    if gamma is None or np.isnan(gamma):
        return np.nan
    return gamma * oi * (S ** 2) * 0.01

###########################################
# GAMMA & GEX VISUALIZATIONS
###########################################
def plot_gamma_heatmap(df):
    st.subheader("Gamma Heatmap by Strike and Time")
    fig_gamma_heatmap = px.density_heatmap(
        df,
        x="date_time",
        y="k",
        z="gamma",
        color_continuous_scale="Viridis",
        title="Gamma by Strike Over Time"
    )
    fig_gamma_heatmap.update_layout(height=400, width=800)
    st.plotly_chart(fig_gamma_heatmap, use_container_width=True)

def plot_gex_by_strike(df_gex):
    st.subheader("Gamma Exposure (GEX) by Strike")
    fig_gex = px.bar(
        df_gex,
        x="strike",
        y="gex",
        color="option_type",
        title="Gamma Exposure (GEX) by Strike",
        labels={"gex": "GEX", "strike": "Strike Price"}
    )
    fig_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_gex, use_container_width=True)

def plot_net_gex(df_gex, spot_price):
    st.subheader("Net Gamma Exposure by Strike")
    df_gex_net = df_gex.groupby("strike").apply(
        lambda x: x.loc[x["option_type"] == "C", "gex"].sum() - x.loc[x["option_type"] == "P", "gex"].sum()
    ).reset_index(name="net_gex")
    df_gex_net["sign"] = df_gex_net["net_gex"].apply(lambda val: "Negative" if val < 0 else "Positive")
    fig_net_gex = px.bar(
        df_gex_net,
        x="strike",
        y="net_gex",
        color="sign",
        color_discrete_map={"Negative": "orange", "Positive": "blue"},
        title="Net Gamma Exposure (Calls GEX - Puts GEX)",
        labels={"net_gex": "Net GEX", "strike": "Strike Price"}
    )
    fig_net_gex.add_hline(y=0, line_dash="dash", line_color="red")
    fig_net_gex.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="lightgrey",
        annotation_text=f"Spot {spot_price:.0f}",
        annotation_position="top right"
    )
    fig_net_gex.update_layout(height=400, width=800)
    st.plotly_chart(fig_net_gex, use_container_width=True)

###########################################
# PERCENTILE-BASED RISK CLASSIFICATIONS
###########################################
def classify_volatility_regime(current_vol, historical_vols):
    percentile = percentileofscore(historical_vols, current_vol)
    if percentile < 5:
        return "Low Volatility"
    elif percentile > 95:
        return "High Volatility"
    else:
        return "Medium Volatility"

def classify_vrp_regime(current_vrp, historical_vrps):
    percentile = percentileofscore(historical_vrps, current_vrp)
    if current_vrp < 0:
        return "Long Volatility"
    elif percentile > 75:
        return "Short Volatility"
    elif percentile < 25:
        return "Long Volatility"
    else:
        return "Neutral"
def calculate_roger_satchell_volatility(price_data, window_days=7, annualize_days=365):
    """
    Calculate Realized Volatility using the Roger-Satchell estimator over a given window in days.
    """
    if price_data.empty or not set(['open', 'high', 'low', 'close']).issubset(price_data.columns):
        return np.nan
    
    if "date_time" in price_data.columns:
        daily_data = price_data.resample('D', on='date_time').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    else:
        daily_data = price_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
    
    if len(daily_data) < window_days:
        return np.nan

    daily_data = daily_data.iloc[-window_days:]
    rs_squared = (
        np.log(daily_data['high'] / daily_data['open']) * np.log(daily_data['high'] / daily_data['close']) +
        np.log(daily_data['low'] / daily_data['open']) * np.log(daily_data['low'] / daily_data['close'])
    )
    daily_rs_vol = np.sqrt(rs_squared.mean())
    annualized_vol = daily_rs_vol * np.sqrt(annualize_days)
    return annualized_vol

###########################################
# FUNCTION TO CALCULATE EXPECTED VALUE (EV) FOR ATM STRADDLE
###########################################
def calculate_atm_straddle_ev(ticker_list, spot_price, T, rv):
    """
    For options within ±2% of the spot, group by strike and average the IV.
    Then compute EV for an ATM straddle using:
        EV = S * (RV - avg_IV) * sqrt(2T/π)
    Returns a DataFrame with candidate strikes, average IV, and EV.
    """
    tolerance = spot_price * 0.02  
    atm_candidates = [item for item in ticker_list if abs(item["strike"] - spot_price) <= tolerance]
    if not atm_candidates:
        return None
    atm_strikes = {}
    for item in atm_candidates:
        strike = item["strike"]
        if strike not in atm_strikes:
            atm_strikes[strike] = {"iv_sum": item["iv"], "count": 1}
        else:
            atm_strikes[strike]["iv_sum"] += item["iv"]
            atm_strikes[strike]["count"] += 1
    ev_candidates = []
    for strike, data in atm_strikes.items():
        avg_iv = data["iv_sum"] / data["count"]
        ev_value = spot_price * (rv - avg_iv) * np.sqrt(2 * T / np.pi)
        ev_candidates.append({"Strike": strike, "Avg IV": avg_iv, "EV": ev_value})
    df_ev = pd.DataFrame(ev_candidates)
    return df_ev.sort_values("EV", ascending=False)

###########################################
# UPDATED TRADE STRATEGY EVALUATION
###########################################
def evaluate_trade_strategy(df, spot_price, risk_tolerance="Moderate", df_iv_agg_reset=None,
                            historical_vols=None, historical_vrps=None, days_to_expiration=7):
    # Corrected: call the properly defined function calculate_roger_satchell_volatility
    rv_vol_series = calculate_roger_satchell_volatility(df_kraken, window_days=7)
    rv_vol = rv_vol_series.iloc[-1] if not rv_vol_series.empty else np.nan
    iv_vol = df["iv_close"].mean() if not df.empty else np.nan

    rv_var = rv_vol ** 2 if not pd.isna(rv_vol) else np.nan
    iv_var = iv_vol ** 2 if not pd.isna(iv_vol) else np.nan
    current_vrp = iv_var - rv_var if (not pd.isna(iv_vol) and not pd.isna(rv_vol)) else np.nan

    divergence = abs(iv_vol - rv_vol) / rv_vol if (rv_vol != 0 and not pd.isna(iv_vol)) else np.nan
    if not np.isnan(divergence) and divergence > 0.20:
        st.write(f"Alert: IV and RV diverge by {divergence*100:.2f}% (threshold: 20%).")

    vol_regime = classify_volatility_regime(rv_vol, historical_vols) if historical_vols and len(historical_vols) > 0 else "Neutral"
    vrp_regime = classify_vrp_regime(current_vrp, historical_vrps) if historical_vrps and len(historical_vrps) > 0 else "Neutral"

    call_items = [item for item in ticker_list if item["option_type"] == "C"]
    put_items = [item for item in ticker_list if item["option_type"] == "P"]
    call_oi_total = sum(item["open_interest"] for item in call_items)
    put_oi_total = sum(item["open_interest"] for item in put_items)
    avg_call_delta = (sum(item["delta"] * item["open_interest"] for item in call_items) / call_oi_total) if call_oi_total > 0 else 0
    avg_put_delta = (sum(item["delta"] * item["open_interest"] for item in put_items) / put_oi_total) if put_oi_total > 0 else 0

    df_ticker = pd.DataFrame(ticker_list) if ticker_list else pd.DataFrame()
    call_oi = df_ticker[df_ticker["option_type"] == "C"]["open_interest"].sum() if not df_ticker.empty else 0
    put_oi = df_ticker[df_ticker["option_type"] == "P"]["open_interest"].sum() if not df_ticker.empty else 0
    put_call_ratio = put_oi / call_oi if call_oi > 0 else np.inf

    df_calls = df[df["option_type"] == "C"].copy()
    df_puts = df[df["option_type"] == "P"].copy()
    if "gamma" not in df_calls.columns:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    if "gamma" not in df_puts.columns:
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    avg_call_gamma = df_calls["gamma"].mean() if not df_calls.empty else 0
    avg_put_gamma = df_puts["gamma"].mean() if not df_puts.empty else 0

    if vrp_regime == "Long Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Long Volatility (Conservative): Buy limited OTM Puts"
            position = "Limited OTM Puts"
            hedge_action = "Hedge lightly with BTC futures short"
        elif risk_tolerance == "Moderate":
            recommendation = "Long Volatility (Neutral): Consider delta-hedged straddles"
            position = "ATM Straddle"
            hedge_action = "Implement dynamic delta hedging"
        else:
            recommendation = "Long Volatility (Aggressive): Take leveraged long volatility positions"
            position = "Leveraged Long Straddle"
            hedge_action = "Aggressively hedge using BTC futures"
    elif vrp_regime == "Short Volatility":
        if risk_tolerance == "Conservative":
            recommendation = "Short Volatility (Conservative): Sell a small number of call spreads"
            position = "Call Spread"
            hedge_action = "Hedge by buying BTC futures"
        elif risk_tolerance == "Moderate":
            recommendation = "Short Volatility (Neutral): Sell strangles with tight stops"
            position = "Strangle"
            hedge_action = "Monitor and adjust hedge dynamically"
        else:
            recommendation = "Short Volatility (Aggressive): Consider naked call selling"
            position = "Naked Calls"
            hedge_action = "Aggressively hedge with BTC futures"
    else:
        recommendation = "Gamma Scalping (Neutral): Consider buying small ATM straddles"
        position = "Small ATM Straddle"
        hedge_action = "Implement light delta hedging"

    return {
        "recommendation": recommendation,
        "position": position,
        "hedge_action": hedge_action,
        "iv": iv_vol,
        "rv": rv_vol,
        "vol_regime": vol_regime,
        "vrp_regime": vrp_regime,
        "put_call_ratio": put_call_ratio,
        "avg_call_delta": avg_call_delta,
        "avg_put_delta": avg_put_delta,
        "avg_call_gamma": avg_call_gamma,
        "avg_put_gamma": avg_put_gamma
    }

###########################################
# MODIFIED MAIN DASHBOARD
###########################################
def main():
    login()  # Ensure user is logged in
    st.title("Crypto Options Visualization Dashboard (Plotly Version) with Volatility Trading Decisions")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # Set expiration date and other sidebar options
    current_date = dt.datetime.now()
    valid_days = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_days)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiration = (expiry_date - current_date).days
    T_YEARS = days_to_expiration / 365
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # Fetch Kraken data (dual timeframe: 5m for real-time and 1d for volatility)
    global df_kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    # Fetch and filter instruments
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts

    # Fetch options data from Thalex API
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return
    
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    # Aggregate IV data for display
    df_iv_agg = (
        df.groupby("date_time", as_index=False)["iv_close"]
        .mean()
        .rename(columns={"iv_close": "iv_mean"})
    )
    df_iv_agg["date_time"] = pd.to_datetime(df_iv_agg["date_time"])
    df_iv_agg = df_iv_agg.set_index("date_time")
    df_iv_agg = df_iv_agg.resample("5T").mean().ffill().sort_index()
    df_iv_agg["rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()
    df_iv_agg["market_regime"] = np.where(
        df_iv_agg["iv_mean"] > df_iv_agg["rolling_mean"], "Risk-Off", "Risk-On"
    )
    df_iv_agg_reset = df_iv_agg.reset_index()

    # Build ticker list for further analysis
    global ticker_list
    ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not (ticker_data and "open_interest" in ticker_data):
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        iv_val = ticker_data.get("iv", None)
        if iv_val is None:
            continue
        T = T_YEARS
        try:
            d1 = (np.log(spot_price / strike) + 0.5 * iv_val**2 * T) / (iv_val * np.sqrt(T))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": ticker_data["open_interest"],
            "delta": delta_est,
            "iv": iv_val
        })
    
    # Compute historical realized volatility and IV metrics
    daily_rv = compute_daily_realized_volatility(df_kraken)
    daily_iv = compute_daily_average_iv(df_iv_agg)
    historical_vrps = compute_historical_vrp(daily_iv, daily_rv)
    
    # Evaluate trade strategy
    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    trade_decision = evaluate_trade_strategy(df, spot_price, risk_tolerance, df_iv_agg_reset,
                                               historical_vols=daily_rv,
                                               historical_vrps=historical_vrps,
                                               days_to_expiration=days_to_expiration)
    
    st.write("### Market and Volatility Metrics")
    st.write(f"Implied Volatility (IV): {trade_decision['iv']:.2%}")
    st.write(f"Realized Volatility (RV): {trade_decision['rv']:.2%}")
    st.write(f"Volatility Regime: {trade_decision['vol_regime']}")
    st.write(f"VRP Regime: {trade_decision['vrp_regime']}")
    st.write(f"Put/Call Open Interest Ratio: {trade_decision['put_call_ratio']:.2f}")
    st.write(f"Average Call Delta: {trade_decision['avg_call_delta']:.4f}")
    st.write(f"Average Put Delta: {trade_decision['avg_put_delta']:.4f}")
    st.write(f"Average Gamma: {trade_decision['avg_call_gamma']:.6f}")
   
    st.write("### Trading Recommendation")
    st.write(f"**Recommendation:** {trade_decision['recommendation']}")
    st.write(f"**Position:** {trade_decision['position']}")
    st.write(f"**Hedge Action:** {trade_decision['hedge_action']}")
    
    # For EV calculations, extract a scalar from the volatility series
    rv_series = calculate_roger_satchell_volatility(df_kraken, window_days=7, annualize_days=365)
    rv_scalar = rv_series.iloc[-1] if (not rv_series.empty and not pd.isna(rv_series).any()) else np.nan
    df_ev = calculate_atm_straddle_ev(ticker_list, spot_price, T_YEARS, rv_scalar)
    if df_ev is not None and not df_ev.empty and not df_ev["EV"].isna().all():
        df_ev_clean = df_ev.dropna(subset=["EV"])
        if not df_ev_clean.empty:
            best_candidate = df_ev_clean.loc[df_ev_clean["EV"].idxmax()]
            best_strike = best_candidate["Strike"]
            st.subheader("ATM Straddle EV Analysis")
            st.write("Candidate Strikes and their Expected Value (EV) in $:")
            st.dataframe(df_ev_clean.style.format({"Avg IV": "{:.2%}", "EV": "{:.2f}"}))
            st.write(f"Recommended ATM Strike based on highest EV: {best_strike}")
        else:
            st.write("No ATM candidates found with valid EV values.")
    else:
        st.write("No ATM candidates found within tolerance for EV calculation.")
    
    if st.button("Simulate Trade"):
        st.write("Simulating trade based on recommendation...")
        st.write("Position Size: Adjust based on capital (e.g., 1-5% of portfolio for chosen risk tolerance)")
        st.write("Monitor price and volatility in real-time and adjust hedges dynamically.")
    
    if not df_calls.empty and not df_puts.empty:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    st.subheader("Volatility Smile at Latest Timestamp")
    latest_ts = df["date_time"].max()
    smile_df = df[df["date_time"] == latest_ts]
    if not smile_df.empty:
        atm_strike = smile_df.loc[smile_df["mark_price_close"].idxmax(), "k"]
        smile_df = smile_df.sort_values(by="k")
        fig_vol_smile = px.line(
            smile_df,
            x="k", y="iv_close",
            markers=True,
            title=f"Volatility Smile at {latest_ts.strftime('%d %b %H:%M')}",
            labels={"iv_close": "IV", "k": "Strike"}
        )
        cheap_hedge_strike = smile_df.loc[smile_df["iv_close"].idxmin(), "k"]
        fig_vol_smile.add_vline(
            x=cheap_hedge_strike,
            line=dict(dash="dash", color="green"),
            annotation_text=f"Cheap Hedge ({cheap_hedge_strike})",
            annotation_position="top"
        )
        fig_vol_smile.add_vline(
            x=spot_price,
            line=dict(dash="dash", color="blue"),
            annotation_text=f"Price: {spot_price:.2f}",
            annotation_position="bottom left"
        )
        fig_vol_smile.update_layout(height=400, width=600)
        st.plotly_chart(fig_vol_smile, use_container_width=True)
        plot_gamma_heatmap(pd.concat([df_calls, df_puts]))
    
    gex_data = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if not (ticker_data and "open_interest" in ticker_data):
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        if option_type == "C":
            candidate = df_calls[df_calls["instrument_name"] == instrument]
        else:
            candidate = df_puts[df_puts["instrument_name"] == instrument]
        if candidate.empty:
            continue
        row = candidate.iloc[0]
        gex = compute_gex(row, spot_price, ticker_data["open_interest"])
        gex_data.append({"strike": strike, "gex": gex, "option_type": option_type})
    df_gex = pd.DataFrame(gex_data)
    if not df_gex.empty:
        plot_gex_by_strike(df_gex)
        plot_net_gex(df_gex, spot_price)

if __name__ == '__main__':
    main()

