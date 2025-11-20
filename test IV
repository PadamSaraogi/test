import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ==============================
# Black–Scholes + IV utilities
# ==============================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no SciPy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black–Scholes price for a European call/put on index/stock with cont. dividend.
    option_type: 'C' or 'P' (case-insensitive).
    """
    option_type = option_type.upper()
    if T <= 0 or sigma <= 0:
        # fallback: intrinsic only
        if option_type == "C":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # forward
    fwd = S * math.exp((r - q) * T)
    vol_sqrtT = sigma * math.sqrt(T)
    try:
        d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
    except ValueError:
        # log domain errors, etc.
        return float("nan")
    d2 = d1 - vol_sqrtT

    if option_type == "C":
        # discounted forward call
        return math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        # put via call–put parity
        call = math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
        return call - math.exp(-r * T) * (fwd - K)


def implied_vol_bs(
    market_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    option_type: str,
    sigma_low: float = 1e-4,
    sigma_high: float = 10.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Robust bisection IV solver.

    - Rejects rows where price < intrinsic or <= 0.
    - Uses wide sigma bounds to allow very high IVs (~1000%+).
    - Returns np.nan if cannot find a solution.
    """
    option_type = option_type.upper()
    intrinsic = max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)

    # Basic sanity checks
    if market_price <= 0 or market_price < intrinsic:
        return np.nan

    # Price at low/high
    p_low = bs_price(S, K, r, q, T, sigma_low, option_type)
    p_high = bs_price(S, K, r, q, T, sigma_high, option_type)

    # If even huge sigma can't reach market_price -> unrealistic
    if np.isnan(p_low) or np.isnan(p_high) or p_high < market_price:
        return np.nan

    low, high = sigma_low, sigma_high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        p_mid = bs_price(S, K, r, q, T, mid, option_type)

        if np.isnan(p_mid):
            return np.nan

        diff = p_mid - market_price
        if abs(diff) < tol:
            return mid

        if diff > 0:
            high = mid
        else:
            low = mid

    return mid  # best effort


# ==============================
# Streamlit App
# ==============================

st.set_page_config(
    page_title="Option Chains, Smiles & IV Surface",
    layout="wide",
)

st.title("Option Chains, Smiles & IV Surface")


# --------- Simple single-option demo (Tab 1) ---------

tab_single, tab_chains = st.tabs(["Single Option Pricer", "Chains, Smiles & IV Surface"])

with tab_single:
    st.subheader("Single Black–Scholes Option")

    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Spot (S)", value=26192.15, min_value=0.0)
        K = st.number_input("Strike (K)", value=26000.0, min_value=0.0)
        option_type = st.selectbox("Type", ["Call (C)", "Put (P)"])
    with col2:
        r = st.number_input("Risk-free rate r (decimal)", value=0.10)
        q = st.number_input("Dividend yield q (decimal)", value=0.01)
    with col3:
        T = st.number_input("Time to maturity T (years)", value=0.10, min_value=0.0)
        sigma = st.number_input("Volatility σ (decimal)", value=0.20, min_value=0.0)

    otype_char = "C" if option_type.startswith("Call") else "P"
    price = bs_price(S, K, r, q, T, sigma, otype_char)
    st.metric("Model price", f"{price:,.4f}")

    target_price = st.number_input("Market price (for IV)", value=price, min_value=0.0)
    if st.button("Solve Implied Vol", key="iv_single"):
        iv = implied_vol_bs(target_price, S, K, r, q, T, otype_char)
        if np.isnan(iv):
            st.error("Could not solve IV (check price vs intrinsic and parameters).")
        else:
            st.success(f"Implied Volatility: {iv:.4%}")


# --------- Chains & Smiles / IV Surface (Tab 2) ---------

with tab_chains:
    st.subheader("Upload Option Chain CSV")

    st.markdown(
        """
        **Expected CSV columns (exact names):**
        - `strike`  
        - `type` (C/P or c/p)  
        - `price` (market option price)  
        - `spot`  
        - `rate` (risk-free, decimal)  
        - `dividend` (q, decimal)  
        - `ttm` (time to maturity in years)  
        - `expiry` (YYYY-MM-DD)
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="chain_csv")

    if uploaded is None:
        st.info("Upload your CSV to see IV smiles and surfaces.")
    else:
        # ---- Read + normalise columns ----
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # standardise column names to lower
        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {"strike", "type", "price", "spot", "rate", "dividend", "ttm", "expiry"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"CSV missing required columns: {', '.join(sorted(missing))}")
            st.stop()

        # parse expiry to datetime
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")

        # clean type column
        df["type"] = df["type"].astype(str).str.strip().str.upper()
        df = df[df["type"].isin(["C", "P"])]

        # numeric types
        for col in ["strike", "price", "spot", "rate", "dividend", "ttm"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["strike", "price", "spot", "rate", "dividend", "ttm", "expiry"])

        if df.empty:
            st.error("No valid rows after cleaning. Check your CSV.")
            st.stop()

        st.write("### Cleaned option chain (all expiries)")
        st.dataframe(df.head(50), use_container_width=True)

        # ---- Expiry selection for IV smile ----
        expiries = sorted(df["expiry"].dropna().unique())
        if len(expiries) == 0:
            st.error("No valid expiry dates parsed from 'expiry' column.")
            st.stop()

        exp_choice = st.selectbox(
            "Select expiry for IV smile",
            options=expiries,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
        )

        df_exp = df[df["expiry"] == exp_choice].copy()
        st.write(f"### Chain for expiry {exp_choice.strftime('%Y-%m-%d')}")
        st.dataframe(df_exp, use_container_width=True)

        # ---- Compute IV for this expiry ----
        st.write("### Computing IV for selected expiry…")
        ivs = []
        for _, row in df_exp.iterrows():
            iv = implied_vol_bs(
                market_price=row["price"],
                S=row["spot"],
                K=row["strike"],
                r=row["rate"],
                q=row["dividend"],
                T=row["ttm"],
                option_type=row["type"],
            )
            ivs.append(iv)

        df_exp["iv"] = ivs
        df_valid = df_exp.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])

        st.write(f"Valid IV rows for this expiry: **{len(df_valid)} / {len(df_exp)}**")
        st.dataframe(df_valid, use_container_width=True)

        # ---- Plot IV smile for this expiry ----
        if df_valid.empty:
            st.warning("No valid IVs solved for this expiry. Check price vs intrinsic, ttm, etc.")
        else:
            smile_chart = (
                alt.Chart(df_valid)
                .mark_line(point=True)
                .encode(
                    x=alt.X("strike:Q", title="Strike"),
                    y=alt.Y("iv:Q", title="Implied Vol (decimal)"),
                    color=alt.Color("type:N", title="Type (C/P)"),
                    tooltip=["strike", "type", "price", "iv"],
                )
                .properties(width=700, height=400, title="IV Smile")
            )
            st.altair_chart(smile_chart, use_container_width=True)

        # ---- IV surface across ALL expiries ----
        st.write("### IV Surface Data (all expiries)")

        iv_all = []
        for _, row in df.iterrows():
            iv = implied_vol_bs(
                market_price=row["price"],
                S=row["spot"],
                K=row["strike"],
                r=row["rate"],
                q=row["dividend"],
                T=row["ttm"],
                option_type=row["type"],
            )
            iv_all.append(iv)

        df["iv"] = iv_all
        df_surface = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])

        st.write(f"Total rows with valid IV: **{len(df_surface)} / {len(df)}**")
        st.dataframe(df_surface.head(50), use_container_width=True)

        # Example 2D slice: IV vs TTM by moneyness bucket
        st.write("#### Example: IV vs TTM at ATM-ish strikes (for quick sanity check)")
        atm_mask = (df_surface["strike"] >= df_surface["spot"] * 0.98) & (
            df_surface["strike"] <= df_surface["spot"] * 1.02
        )
        df_atm = df_surface[atm_mask].copy()
        if not df_atm.empty:
            atm_chart = (
                alt.Chart(df_atm)
                .mark_point()
                .encode(
                    x=alt.X("ttm:Q", title="TTM (years)"),
                    y=alt.Y("iv:Q", title="ATM Implied Vol"),
                    color="type:N",
                    tooltip=["expiry", "strike", "iv"],
                )
                .properties(width=700, height=350)
            )
            st.altair_chart(atm_chart, use_container_width=True)
        else:
            st.info("No near-ATM points found for the surface preview.")
