# ============================================================
# arima_forecaster.py
# Modul ARIMA untuk ForecastIQ — drop-in module
# Cara pakai: import file ini, panggil run_arima_comparison()
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product


# ────────────────────────────────────────────────
# 1. AUTO ORDER SELECTION (grid search AIC)
# ────────────────────────────────────────────────
def _best_arima_order(data: list, max_p=3, max_d=2, max_q=3) -> tuple:
    """Cari (p,d,q) terbaik berdasarkan AIC."""
    best_aic   = np.inf
    best_order = (1, 1, 1)

    for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(data, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic   = res.aic
                best_order = (p, d, q)
        except Exception:
            continue

    return best_order, best_aic


# ────────────────────────────────────────────────
# 2. HELPERS
# ────────────────────────────────────────────────
def _mape(actual, predicted) -> float:
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    mask = a != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def _adf_test(data: list) -> tuple:
    """Return (stationary: bool, p_value: float)"""
    if len(data) < 5:
        return True, 0.0
    try:
        res   = adfuller(data, autolag='AIC')
        p_val = res[1]
        return p_val < 0.05, round(p_val, 4)
    except Exception:
        return True, 0.0


# ────────────────────────────────────────────────
# 3. MAIN — panggil dari halaman "Hasil Forecasting"
# ────────────────────────────────────────────────
def run_arima_comparison(
    demand_data: list,
    ann_forecast: list,
    ann_mape: float,
    n_forecast: int = 8,
):
    """
    Tampilkan perbandingan ARIMA vs ANN di bawah hasil ANN.

    Parameter
    ---------
    demand_data  : list — data historis demand dari input user
    ann_forecast : list — hasil prediksi ANN (sudah ada di sistem)
    ann_mape     : float — nilai MAPE ANN (sudah ada di sistem)
    n_forecast   : int  — jumlah bulan ke depan (default 8)

    Cara integrasi
    --------------
    Pada file halaman "Hasil Forecasting" Anda, setelah blok
    menampilkan hasil ANN, tambahkan:

        from arima_forecaster import run_arima_comparison

        run_arima_comparison(
            demand_data  = st.session_state["demand_list"],
            ann_forecast = st.session_state["ann_predictions"],
            ann_mape     = st.session_state["ann_mape"],
            n_forecast   = st.session_state["n_forecast"],
        )
    """

    st.markdown("---")
    st.markdown("## 🔁 Pembanding Otomatis: ARIMA vs ANN")
    st.caption(
        "ARIMA dijalankan otomatis dengan data yang sama. "
        "Order (p,d,q) dipilih secara otomatis via grid search AIC."
    )

    # ── Validasi ──
    if len(demand_data) < 6:
        st.warning(
            "⚠️ Data terlalu sedikit untuk ARIMA (minimal 6 titik data). "
            "Tambahkan lebih banyak data historis."
        )
        return

    data = [float(x) for x in demand_data]

    # ── Cache order agar tidak re-run setiap render ──
    cache_key = f"arima_order_{hash(str(data))}"
    if cache_key not in st.session_state:
        with st.spinner("🔍 Mencari order ARIMA terbaik (p,d,q) via AIC..."):
            order, aic = _best_arima_order(data)
        st.session_state[cache_key] = (order, aic)
    else:
        order, aic = st.session_state[cache_key]

    p, d, q = order

    # ── Fit model ──
    try:
        model  = ARIMA(data, order=order)
        result = model.fit()
    except Exception as e:
        st.error(f"❌ ARIMA gagal: {e}")
        st.info("💡 Coba tambah data historis (minimal 10 titik data ideal untuk ARIMA).")
        return

    # ── MAPE in-sample ──
    fitted       = list(result.fittedvalues)
    actual_trim  = data[d:] if d > 0 else data
    fitted_trim  = fitted[d:] if d > 0 else fitted
    min_len      = min(len(actual_trim), len(fitted_trim))
    arima_mape   = _mape(actual_trim[:min_len], fitted_trim[:min_len])

    # ── Forecast ──
    fc_obj       = result.get_forecast(steps=n_forecast)
    arima_fc     = list(fc_obj.predicted_mean)
    conf         = fc_obj.conf_int()
    lower_ci = list(conf[:, 0])
    upper_ci = list(conf[:, 1])

    # ── ADF test ──
    stationary, p_val = _adf_test(data)

    # ════════════════════════════════════════
    # TAMPILAN
    # ════════════════════════════════════════

    # ── Info model ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Order ARIMA",     f"({p},{d},{q})")
    c2.metric("AIC Score",       f"{aic:.2f}")
    c3.metric("Data Stasioner",  "Ya ✅" if stationary else f"Tidak → d={d}")
    c4.metric("ADF p-value",     str(p_val))

    st.markdown("---")

    # ── Tabel + Chart ──
    col_tbl, col_chart = st.columns([1, 2])

    with col_tbl:
        st.markdown("#### 📋 Prediksi ARIMA")
        df = pd.DataFrame({
            "Bulan"    : [f"+{i+1}" for i in range(n_forecast)],
            "Prediksi" : [round(v) for v in arima_fc],
            "Lower CI" : [round(l) for l in lower_ci],
            "Upper CI" : [round(u) for u in upper_ci],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown(f"**Rata-rata:** {np.mean(arima_fc):,.0f}")
        st.markdown(f"**Maks:** {max(arima_fc):,.0f}")
        st.markdown(f"**Min:** {min(arima_fc):,.0f}")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV ARIMA",
            data=csv,
            file_name="forecast_arima.csv",
            mime="text/csv",
        )

    with col_chart:
        st.markdown("#### 📈 Grafik Perbandingan")

        n_hist = len(data)
        x_hist = list(range(n_hist))
        x_fore = list(range(n_hist, n_hist + n_forecast))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#111118")
        ax.set_facecolor("#111118")

        # Historis
        ax.plot(
            x_hist, data,
            color="#e2e2ee", linewidth=2,
            marker="o", markersize=4,
            label="Data Historis", zorder=3
        )

        # ANN forecast
        if len(ann_forecast) >= n_forecast:
            ax.plot(
                x_fore, ann_forecast[:n_forecast],
                color="#00c8a8", linewidth=2, linestyle="--",
                marker="s", markersize=4,
                label=f"ANN  — MAPE {ann_mape:.2f}%", zorder=3
            )

        # ARIMA forecast
        ax.plot(
            x_fore, arima_fc,
            color="#e63950", linewidth=2, linestyle="--",
            marker="^", markersize=4,
            label=f"ARIMA{order} — MAPE {arima_mape:.2f}%", zorder=3
        )

        # Confidence interval
        ax.fill_between(
            x_fore, lower_ci, upper_ci,
            color="#e63950", alpha=0.12,
            label="95% CI ARIMA"
        )

        # Batas historis
        ax.axvline(x=n_hist - 0.5, color="#444466", linestyle=":", linewidth=1.5)
        ylim = ax.get_ylim()
        ax.text(
            n_hist - 0.3, ylim[1] * 0.98,
            "Batas Historis", color="#6a6a88",
            fontsize=7, va="top"
        )

        ax.set_xlabel("Periode", color="#9898b8", fontsize=9)
        ax.set_ylabel("Demand",  color="#9898b8", fontsize=9)
        ax.tick_params(colors="#9898b8", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3e")
        ax.grid(color="#1e1e2e", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.legend(
            facecolor="#17171f", edgecolor="#2a2a3e",
            labelcolor="#e2e2ee", fontsize=8
        )
        ax.set_title(
            "ForecastIQ — ANN vs ARIMA",
            color="#e2e2ee", fontsize=10, pad=10
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── Scorecard ──
    st.markdown("#### 🏆 Scorecard Model")

    winner = "ARIMA" if arima_mape < ann_mape else "ANN"
    diff   = abs(ann_mape - arima_mape)
    w_clr  = "#e63950" if winner == "ARIMA" else "#00c8a8"

    ca, cv, cb = st.columns([2, 1, 2])

    with ca:
        border = f"2px solid #00c8a8" if winner == "ANN" else "1px solid #2a2a3e"
        crown  = "<div style='font-size:.72rem;color:#00c8a8;margin-top:.5rem'>👑 Lebih Akurat</div>" if winner == "ANN" else ""
        st.markdown(f"""
        <div style="border:{border};padding:1.25rem;text-align:center;background:#17171f;border-radius:4px;">
          <div style="font-family:monospace;font-size:.62rem;color:#6a6a88;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.4rem">ANN</div>
          <div style="font-family:monospace;font-size:2rem;font-weight:700;color:#00c8a8">{ann_mape:.2f}%</div>
          <div style="font-size:.72rem;color:#9898b8;margin-top:.25rem">MAPE</div>
          {crown}
        </div>""", unsafe_allow_html=True)

    with cv:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:100%;padding-top:.8rem;
                    font-family:monospace;font-size:1.1rem;color:#6a6a88">
          VS
        </div>""", unsafe_allow_html=True)

    with cb:
        border = f"2px solid #e63950" if winner == "ARIMA" else "1px solid #2a2a3e"
        crown  = "<div style='font-size:.72rem;color:#e63950;margin-top:.5rem'>👑 Lebih Akurat</div>" if winner == "ARIMA" else ""
        st.markdown(f"""
        <div style="border:{border};padding:1.25rem;text-align:center;background:#17171f;border-radius:4px;">
          <div style="font-family:monospace;font-size:.62rem;color:#6a6a88;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.4rem">ARIMA {order}</div>
          <div style="font-family:monospace;font-size:2rem;font-weight:700;color:#e63950">{arima_mape:.2f}%</div>
          <div style="font-size:.72rem;color:#9898b8;margin-top:.25rem">MAPE</div>
          {crown}
        </div>""", unsafe_allow_html=True)

    # Kesimpulan
    if winner == "ARIMA":
        kesimpulan = "ARIMA bekerja lebih baik — pola data cenderung linear/stasioner sehingga model statistik klasik lebih cocok."
    else:
        kesimpulan = "ANN menangkap pola non-linear lebih baik — data demand memiliki fluktuasi kompleks yang lebih cocok untuk neural network."

    st.markdown(f"""
    <div style="margin-top:1rem;padding:.85rem 1rem;background:#17171f;
                border-left:3px solid {w_clr};font-size:.85rem;color:#9898b8;border-radius:0 4px 4px 0">
      <strong style="color:#e2e2ee">Kesimpulan:</strong>
      Model <strong style="color:{w_clr}">{winner}</strong> lebih akurat
      dengan selisih MAPE <strong style="color:{w_clr}">{diff:.2f}%</strong>. {kesimpulan}
    </div>""", unsafe_allow_html=True)

    # ── Detail statistik (collapsible) ──
    st.markdown("---")
    with st.expander("🔬 Detail Statistik ARIMA"):
        d1, d2, d3 = st.columns(3)
        d1.metric("Rata-rata Forecast", f"{np.mean(arima_fc):,.0f}")
        d2.metric("Maks Forecast",      f"{max(arima_fc):,.0f}")
        d3.metric("Min Forecast",       f"{min(arima_fc):,.0f}")

        st.markdown("**Model Summary (ringkas):**")
        summary_lines = str(result.summary()).split("\n")
        st.code("\n".join(summary_lines[:28]), language="text")
