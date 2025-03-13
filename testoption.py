import streamlit as st
import datetime as dt
import pandas as pd
import requests
import numpy as np
import ccxt
from toolz.curried import *
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
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
# EXPIRATION DATE HELPER FUNCTIONS
###########################################
def get_valid_expiration_options(current_date):
    """
    Returns a list of valid expiration dates using only 14 and 28 days from now.
    """
    options = []
    for days in [14, 28]:
        exp_date = current_date + dt.timedelta(days=days)
        options.append(exp_date.strftime("%d%b%y").upper())
    return options

def compute_expiry_date(selected_day, current_date):
    """
    Converts a selected expiration day string into a datetime object.
    """
    try:
        return dt.datetime.strptime(selected_day, "%d%b%y")
    except Exception:
        return None

###########################################
# CREDENTIALS & LOGIN FUNCTIONS (from text files)
###########################################
def load_credentials():
    """
    Load user credentials from two text files:
    - usernames.txt: one username per line
    - passwords.txt: one password per line (order corresponds to usernames)
    Returns a dictionary mapping username to password.
    """
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
    """
    Displays a login form and validates the credentials.
    The dashboard will only load after successful authentication.
    """
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
    """Fetch the instruments list from the Thalex API."""
    response = requests.get(url_instruments)
    if response.status_code != 200:
        raise Exception("Failed to fetch instruments")
    data = response.json()
    return data.get("result", [])

def get_option_instruments(instruments, option_type):
    """
    Filter instruments for options (calls or puts) for BTC with expiry (using default expiry).
    Option type should be 'C' for calls or 'P' for puts.
    """
    filtered = [
        inst["instrument_name"] for inst in instruments
        if inst["instrument_name"].startswith("BTC-" + DEFAULT_EXPIRY_STR) and inst["instrument_name"].endswith(f"-{option_type}")
    ]
    return sorted(filtered)

def get_actual_iv(instrument_name):
    """
    Fetch mark price data for the given instrument and return its latest iv_close value.
    """
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
    """
    Filter instruments based on the theoretical range derived from a standard deviation move.
    """
    instruments_list = fetch_instruments()
    calls_all = get_option_instruments(instruments_list, "C")
    puts_all = get_option_instruments(instruments_list, "P")
    
    strike_list = [(inst, int(inst.split("-")[2])) for inst in calls_all]
    strike_list.sort(key=lambda x: x[1])
    strikes = [s for _, s in strike_list]
    closest_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    nearest_instrument = strike_list[closest_index][0]
    
    actual_iv = get_actual_iv(nearest_instrument)
    if actual_iv is None:
        raise Exception("Could not fetch actual IV for the nearest instrument")
    
    lower_bound = spot_price * np.exp(-actual_iv * np.sqrt(t_years) * multiplier)
    upper_bound = spot_price * np.exp(actual_iv * np.sqrt(t_years) * multiplier)
    
    filtered_calls = [inst for inst in calls_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    filtered_puts = [inst for inst in puts_all if lower_bound <= int(inst.split("-")[2]) <= upper_bound]
    
    filtered_calls.sort(key=lambda x: int(x.split("-")[2]))
    filtered_puts.sort(key=lambda x: int(x.split("-")[2]))
    return filtered_calls, filtered_puts

###########################################
# DATA FETCHING FUNCTIONS
###########################################
@st.cache_data(ttl=30)
def fetch_data(instruments_tuple):
    """
    Fetches Thalex mark price data for the provided instruments over the past 7 days at 5m resolution.
    """
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
    """
    Fetch ticker data for the given instrument.
    """
    params = {"instrument_name": instrument_name}
    response = requests.get(URL_TICKER, params=params)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("result", {})

def fetch_kraken_data():
    """
    Fetch 7 days of 5m BTC/USD data from Kraken (via ccxt).
    """
    kraken = ccxt.kraken()
    now_dt = dt.datetime.now()
    start_dt = now_dt - dt.timedelta(days=7)
    since = int(start_dt.timestamp() * 1000)
    ohlcv = kraken.fetch_ohlcv("BTC/USD", timeframe="5m", since=since, limit=3000)
    df_kraken = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df_kraken.empty:
        return pd.DataFrame()
    df_kraken["date_time"] = pd.to_datetime(df_kraken["timestamp"], unit="ms")
    df_kraken["date_time"] = df_kraken["date_time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    df_kraken = df_kraken.sort_values(by="date_time").reset_index(drop=True)
    cutoff_start = (now_dt - dt.timedelta(days=7)).astimezone(df_kraken["date_time"].dt.tz)
    df_kraken = df_kraken[df_kraken["date_time"] >= cutoff_start]
    return df_kraken

###########################################
# COMPUTE ROLLING VRP FUNCTION
###########################################
def compute_rolling_vrp(group, window_str):
    """
    Computes rolling variance risk premium (VRP) for a given group over the specified window.
    VRP = (rolling average of squared IV) - (rolling sum of squared log returns)
    """
    rolling_rv = group["log_return"].expanding(min_periods=1).apply(lambda x: np.nansum(x**2), raw=True)
    rolling_iv = group["iv_close"].rolling(window_str, min_periods=1).apply(lambda x: np.mean(x**2), raw=True)
    return rolling_iv - rolling_rv

###########################################
# OPTION DELTA CALCULATION FUNCTION
###########################################
def compute_delta(row, S):
    """
    Compute the Black-Scholes delta for an option.
    """
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
    """
    Compute average call and put delta for each timestamp.
    """
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

def rolling_percentile_zones(df, column="delta_diff", window="1D", lower_percentile=30, upper_percentile=70):
    """
    Compute rolling percentile zones for the specified column.
    """
    df = df.set_index("date_time").sort_index()
    def percentile_in_window(x, q):
        return np.percentile(x, q)
    df["rolling_lower_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, lower_percentile), raw=False)
    df["rolling_upper_zone"] = df[column].rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x, upper_percentile), raw=False)
    return df.reset_index()

def classify_regime(row):
    """
    Classify regime based on delta_diff relative to rolling zones.
    """
    if pd.isna(row["rolling_lower_zone"]) or pd.isna(row["rolling_upper_zone"]):
        return "Neutral"
    if row["delta_diff"] > row["rolling_upper_zone"]:
        return "Risk-On"
    elif row["delta_diff"] < row["rolling_lower_zone"]:
        return "Risk-Off"
    else:
        return "Neutral"

def compute_gamma(row, S):
    """
    Compute the Black-Scholes gamma for an option.
    """
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
    """
    Compute Gamma Exposure (GEX) for an option.
    """
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
# NEW HELPER FUNCTIONS FOR DECISION-MAKING
###########################################
def calculate_realized_volatility(price_data, window_days=7):
    """
    Calculate realized volatility (annualized) from price data over a specified window.
    Uses log returns from the "close" column and annualizes by sqrt(252) trading days.
    """
    if price_data.empty:
        return np.nan
    price_data = price_data.sort_values("date_time")
    log_returns = np.log(price_data["close"] / price_data["close"].shift(1))
    daily_vol = log_returns.rolling(window=int(window_days * 24 * 12), min_periods=1).std()  # ~336 intervals/day
    annualized_vol = daily_vol * np.sqrt(252)
    return annualized_vol.iloc[-1] if not annualized_vol.empty else np.nan

def compare_volatility(iv, rv, threshold=0.1):
    """
    Compare implied volatility (IV) to realized volatility (RV) to recommend volatility direction.
    Returns "Vol Up" if IV < RV - threshold, "Vol Down" if IV > RV + threshold, "Neutral" otherwise.
    """
    if pd.isna(iv) or pd.isna(rv):
        return "Neutral"
    if iv < rv - threshold:
        return "Vol Up"
    elif iv > rv + threshold:
        return "Vol Down"
    else:
        return "Neutral"

def evaluate_trade_strategy(df, spot_price, risk_tolerance="Moderate", df_iv_agg_reset=None):
    """
    Evaluate market conditions and recommend a volatility trading strategy.
    Uses IV, RV, delta, gamma, and open interest to suggest long vol, short vol, or gamma scalping.
    The df_iv_agg_reset parameter is required for market regime analysis.
    """
    rv = calculate_realized_volatility(df_kraken)
    iv = df["iv_close"].mean() if not df.empty else np.nan
    vol_direction = compare_volatility(iv, rv)
    latest_regime = (df_iv_agg_reset["market_regime"].iloc[-1]
                     if (df_iv_agg_reset is not None and "market_regime" in df_iv_agg_reset.columns and not df_iv_agg_reset.empty)
                     else "Neutral")
    
    df_calls = df[df["option_type"] == "C"].copy()
    df_puts = df[df["option_type"] == "P"].copy()
    if "delta" not in df_calls.columns:
        df_calls["delta"] = df_calls.apply(lambda row: compute_delta(row, spot_price), axis=1)
    if "delta" not in df_puts.columns:
        df_puts["delta"] = df_puts.apply(lambda row: compute_delta(row, spot_price), axis=1)
    if "gamma" not in df_calls.columns:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    if "gamma" not in df_puts.columns:
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
    
    avg_call_delta = df_calls["delta"].mean() if not df_calls.empty else 0
    avg_put_delta = df_puts["delta"].mean() if not df_puts.empty else 0
    avg_call_gamma = df_calls["gamma"].mean() if not df_calls.empty else 0
    avg_put_gamma = df_puts["gamma"].mean() if not df_puts.empty else 0
    
    df_ticker = pd.DataFrame(ticker_list) if ticker_list else pd.DataFrame()
    total_oi = df_ticker["open_interest"].sum() if not df_ticker.empty else 0
    put_oi = df_ticker[df_ticker["option_type"] == "P"]["open_interest"].sum() if not df_ticker.empty else 0
    call_oi = df_ticker[df_ticker["option_type"] == "C"]["open_interest"].sum() if not df_ticker.empty else 0
    put_call_ratio = put_oi / call_oi if call_oi > 0 else np.nan
    
    recommendation = "Neutral"
    position = None
    hedge_action = None
    
    if risk_tolerance == "Aggressive":
        if vol_direction == "Vol Up" or latest_regime == "Risk-Off":
            recommendation = "Long Volatility"
            position = "Buy Straddle/Strangle (Calls & Puts)"
            hedge_action = "Delta hedge by selling BTC futures"
        elif vol_direction == "Vol Down" or latest_regime == "Risk-On":
            recommendation = "Short Volatility"
            position = "Sell Straddle/Spread (Calls & Puts)"
            hedge_action = "Monitor for volatility spikes; prepare to buy back if IV rises"
        else:
            recommendation = "Gamma Scalping"
            position = "Buy ATM Straddle, maintain Delta-neutral"
            hedge_action = "Dynamically hedge using BTC futures"
    else:  # Moderate or Conservative
        if vol_direction == "Vol Up" and put_call_ratio > 1 and latest_regime == "Risk-Off":
            recommendation = "Long Volatility (Conservative)"
            position = "Buy OTM Puts"
            hedge_action = "Limit position size; hedge lightly with BTC futures short"
        elif vol_direction == "Vol Down" and put_call_ratio < 1 and latest_regime == "Risk-On":
            recommendation = "Short Volatility (Conservative)"
            position = "Sell OTM Calls"
            hedge_action = "Limit exposure; hedge with small BTC futures long"
        else:
            recommendation = "Gamma Scalping (Conservative)"
            position = "Buy small ATM Straddle"
            hedge_action = "Light delta hedging and monitor closely"
    
    return {
        "recommendation": recommendation,
        "position": position,
        "hedge_action": hedge_action,
        "iv": iv,
        "rv": rv,
        "vol_direction": vol_direction,
        "market_regime": latest_regime,
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
    # Login Section
    login()  # Ensure user is logged in
    
    st.title("Crypto Options Visualization Dashboard (Plotly Version) with Volatility Trading Decisions")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.stop()
    
    # EXPIRATION DATE SELECTION
    current_date = dt.datetime.now()
    valid_options = get_valid_expiration_options(current_date)
    selected_day = st.sidebar.selectbox("Choose Expiration Day", options=valid_options)
    expiry_date = compute_expiry_date(selected_day, current_date)
    if expiry_date is None or expiry_date < current_date:
        st.error("Expiration date is invalid or already passed")
        st.stop()
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    days_to_expiry = (expiry_date - current_date).days
    T_YEARS = days_to_expiry / 365
    st.sidebar.markdown(f"**Using Expiration Date:** {expiry_str}")
    
    # DEVIATION RANGE SELECTION
    deviation_option = st.sidebar.select_slider(
        "Choose Deviation Range",
        options=["1 Standard Deviation (68.2%)", "2 Standard Deviations (95.4%)"],
        value="1 Standard Deviation (68.2%)"
    )
    multiplier = 1 if "1 Standard" in deviation_option else 2

    # Fetch Kraken data
    global df_kraken
    df_kraken = fetch_kraken_data()
    if df_kraken.empty:
        st.error("No data fetched from Kraken. Check your ccxt config or timeframe.")
        return
    spot_price = df_kraken["close"].iloc[-1]
    st.write(f"Current BTC/USD Price: {spot_price:.2f}")
    
    try:
        filtered_calls, filtered_puts = get_filtered_instruments(spot_price, expiry_str, T_YEARS, multiplier)
    except Exception as e:
        st.error(f"Error fetching instruments: {e}")
        return
    st.write("Filtered Call Instruments:", filtered_calls)
    st.write("Filtered Put Instruments:", filtered_puts)
    all_instruments = filtered_calls + filtered_puts
    
    df = fetch_data(tuple(all_instruments))
    if df.empty:
        st.error("No data fetched from Thalex. Please check the API or instrument names.")
        return
    
    df_calls = df[df["option_type"] == "C"].copy().sort_values("date_time")
    df_puts = df[df["option_type"] == "P"].copy().sort_values("date_time")
    
    # Compute aggregated IV for regime analysis
    df_iv_agg = (
        df.groupby("date_time", as_index=False)["iv_close"]
        .mean()
        .rename(columns={"iv_close": "iv_mean"})
    )
    df_iv_agg["date_time"] = pd.to_datetime(df_iv_agg["date_time"])
    df_iv_agg = df_iv_agg.set_index("date_time")
    df_iv_agg = df_iv_agg.resample("5T").mean().ffill()

    # Compute the rolling mean on the datetime-indexed DataFrame
    df_iv_agg = df_iv_agg.sort_index()  # ensure it's sorted by date_time
    df_iv_agg["rolling_mean"] = df_iv_agg["iv_mean"].rolling("1D").mean()

    df_iv_agg["market_regime"] = np.where(
        df_iv_agg["iv_mean"] > df_iv_agg["rolling_mean"], "Risk-Off", "Risk-On"
    )

    # Reset the index for later use if needed
    df_iv_agg_reset = df_iv_agg.reset_index()

    # Create a simple market regime column for decision-making:
    df_iv_agg_reset = df_iv_agg.reset_index()  # date_time becomes a column now
    df_iv_agg_reset["rolling_mean"] = df_iv_agg_reset.rolling("1D", on="date_time")["iv_mean"].mean()

    df_iv_agg_reset["market_regime"] = np.where(
        df_iv_agg_reset["iv_mean"] > df_iv_agg_reset["rolling_mean"], "Risk-Off", "Risk-On"
    )
    
    # Build ticker_list for open interest and delta analysis
    global ticker_list
    ticker_list = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if ticker_data and "open_interest" in ticker_data:
            oi = ticker_data["open_interest"]
        else:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        iv_val = ticker_data.get("iv", None)
        if iv_val is None:
            continue
        T_est = 0.05
        try:
            d1 = (np.log(spot_price / strike) + 0.5 * iv_val**2 * T_est) / (iv_val * np.sqrt(T_est))
        except Exception:
            continue
        delta_est = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1
        ticker_list.append({
            "instrument": instrument,
            "strike": strike,
            "option_type": option_type,
            "open_interest": oi,
            "delta": delta_est
        })
    
    # NEW: VOLATILITY TRADING DECISION TOOL
    st.subheader("Volatility Trading Decision Tool")
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    trade_decision = evaluate_trade_strategy(df, spot_price, risk_tolerance, df_iv_agg_reset)
    
    st.write("### Market and Volatility Metrics")
    st.write(f"Implied Volatility (IV): {trade_decision['iv']:.2%}")
    st.write(f"Realized Volatility (RV): {trade_decision['rv']:.2%}")
    st.write(f"Volatility Direction: {trade_decision['vol_direction']}")
    st.write(f"Market Regime: {trade_decision['market_regime']}")
    st.write(f"Put/Call Open Interest Ratio: {trade_decision['put_call_ratio']:.2f}")
    st.write(f"Average Call Delta: {trade_decision['avg_call_delta']:.4f}")
    st.write(f"Average Put Delta: {trade_decision['avg_put_delta']:.4f}")
    # Display gamma values with higher precision
    st.write(f"Average Call Gamma: {trade_decision['avg_call_gamma']:.6f}")
    st.write(f"Average Put Gamma: {trade_decision['avg_put_gamma']:.6f}")
    
    st.write("### Trading Recommendation")
    st.write(f"**Recommendation:** {trade_decision['recommendation']}")
    st.write(f"**Position:** {trade_decision['position']}")
    st.write(f"**Hedge Action:** {trade_decision['hedge_action']}")
    
    if st.button("Simulate Trade"):
        st.write("Simulating trade based on recommendation...")
        st.write("Position Size: Adjust based on capital (e.g., 1-5% of portfolio for chosen risk tolerance)")
        st.write("Monitor price and volatility in real-time and adjust hedges dynamically.")
    
    # Gamma and GEX visualizations
    if not df_calls.empty and not df_puts.empty:
        df_calls["gamma"] = df_calls.apply(lambda row: compute_gamma(row, spot_price), axis=1)
        df_puts["gamma"] = df_puts.apply(lambda row: compute_gamma(row, spot_price), axis=1)
        plot_gamma_heatmap(pd.concat([df_calls, df_puts]))
    
    gex_data = []
    for instrument in all_instruments:
        ticker_data = fetch_ticker(instrument)
        if ticker_data and "open_interest" in ticker_data:
            oi = ticker_data["open_interest"]
        else:
            continue
        try:
            strike = int(instrument.split("-")[2])
        except Exception:
            continue
        option_type = instrument.split("-")[-1]
        if option_type == "C":
            row = df_calls[df_calls["instrument_name"] == instrument].iloc[0]
        else:
            row = df_puts[df_puts["instrument_name"] == instrument].iloc[0]
        gex = compute_gex(row, spot_price, oi)
        gex_data.append({"strike": strike, "gex": gex, "option_type": option_type})
    df_gex = pd.DataFrame(gex_data)
    if not df_gex.empty:
        plot_gex_by_strike(df_gex)
        plot_net_gex(df_gex, spot_price)

if __name__ == '__main__':
    main()
