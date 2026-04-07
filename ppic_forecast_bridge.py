"""
ppic_forecast_bridge.py
Tambahkan file ini ke folder PPIC Forecasting System kamu.
Fungsi di sini dipanggil setelah model selesai training & menghasilkan prediksi.

Cara pakai di app PPIC kamu (contoh di halaman Hasil Forecasting):

    from ppic_forecast_bridge import save_forecast_results, show_sync_button

    # Setelah forecast selesai:
    if st.button("Mulai Forecasting"):
        hasil = model.predict(...)          # hasil prediksi kamu
        save_forecast_results(
            sku          = selected_sku,    # SKU barang yang di-forecast
            product_name = selected_name,
            predictions  = hasil,           # list angka prediksi per bulan
            mape         = mape_value,
            window_size  = window_size,
        )

    show_sync_button()  # tampilkan tombol "Lihat di Inventory Dashboard"
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text


# ──────────────────────────────────────────
# Koneksi ke database yang SAMA dengan Inventory Dashboard
# Pastikan secrets.toml di project PPIC kamu juga punya [database]
# ──────────────────────────────────────────
@st.cache_resource
def get_shared_engine():
    """
    Gunakan database yang sama dengan Inventory Dashboard.
    Mendukung dua format secrets.toml:
      Format A (PPIC): [database] url = "postgresql://..."
      Format B (Inventory): [database] host/port/user/password/dbname
    """
    try:
        db = st.secrets["database"]
        # Format A — pakai URL langsung (format PPIC Forecasting)
        if "url" in db:
            url = db["url"]
        # Format B — pakai host/user/password terpisah (format Inventory)
        else:
            url = (
                f"postgresql://{db['user']}:{db['password']}"
                f"@{db['host']}:{db['port']}/{db['dbname']}"
            )
        return create_engine(url, pool_pre_ping=True)
    except Exception as e:
        st.warning(f"Tidak bisa konek ke shared database: {e}")
        return None


def save_forecast_results(
    sku: str,
    product_name: str,
    predictions: list,
    mape: float,
    window_size: int,
) -> bool:
    """
    Simpan hasil forecast ke tabel forecast_results di PostgreSQL.
    Dipanggil setelah model menghasilkan prediksi.

    Parameter:
        sku          : SKU barang (harus cocok dengan tabel products di inventory)
        product_name : Nama barang
        predictions  : list angka prediksi per bulan, misal [2463, 2427, 2425, ...]
        mape         : nilai MAPE model (%)
        window_size  : window size yang dipakai

    Contoh pemanggilan:
        save_forecast_results(
            sku          = "BM-001",
            product_name = "Plat Besi 3mm",
            predictions  = [450, 470, 460, 455, 465],
            mape         = 25.5,
            window_size  = 6,
        )
    """
    engine = get_shared_engine()
    if engine is None:
        st.warning("Hasil forecast tidak bisa disimpan ke inventory database.")
        return False

    rows = [
        {
            "product_sku":    sku,
            "product_name":   product_name,
            "bulan_ke":       i + 1,
            "prediksi_demand": float(pred),
            "model_mape":     float(mape),
            "window_size":    int(window_size),
        }
        for i, pred in enumerate(predictions)
    ]

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO forecast_results
                    (product_sku, product_name, bulan_ke,
                     prediksi_demand, model_mape, window_size)
                VALUES
                    (:product_sku, :product_name, :bulan_ke,
                     :prediksi_demand, :model_mape, :window_size)
            """), rows)
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan forecast: {e}")
        return False


def get_sku_list() -> dict:
    """
    Ambil daftar SKU dari tabel products inventory.
    Berguna untuk dropdown pilih barang di PPIC Forecasting.

    Return dict: {"Nama Barang (SKU)": "SKU"}
    """
    engine = get_shared_engine()
    if engine is None:
        return {}
    try:
        df = pd.read_sql(
            "SELECT sku, name FROM products WHERE is_active=TRUE ORDER BY name",
            engine
        )
        return dict(zip(df["name"] + " (" + df["sku"] + ")", df["sku"]))
    except Exception:
        return {}


def show_sync_status() -> None:
    """
    Tampilkan info singkat di halaman PPIC setelah forecast tersimpan.
    Tambahkan ini di bawah tabel hasil forecast.
    """
    engine = get_shared_engine()
    if engine is None:
        return
    try:
        df = pd.read_sql(
            "SELECT COUNT(*) AS total FROM forecast_results "
            "WHERE forecast_date >= NOW() - INTERVAL '1 hour'",
            engine
        )
        total = int(df["total"].iloc[0])
        if total > 0:
            st.success(
                f"✅ {total} baris forecast tersimpan ke Inventory Database. "
                "Buka **Inventory Dashboard → Sync Forecast** untuk mengupdate demand."
            )
    except Exception:
        pass


# ──────────────────────────────────────────
# Contoh integrasi lengkap di halaman PPIC
# ──────────────────────────────────────────
CONTOH_INTEGRASI = """
# ── Tempelkan kode ini di halaman "Hasil Forecasting" di PPIC kamu ──

from ppic_forecast_bridge import save_forecast_results, show_sync_status

# ... kode forecast kamu yang sudah ada ...

# Setelah prediksi berhasil dibuat:
if forecast_berhasil:

    # Simpan ke shared database
    ok = save_forecast_results(
        sku          = sku_barang,        # misal "BM-001"
        product_name = nama_barang,       # misal "Plat Besi 3mm"
        predictions  = list_prediksi,     # misal [2463, 2427, 2425, 2425, 2425]
        mape         = nilai_mape,        # misal 25.50
        window_size  = window_size,       # misal 6
    )

    if ok:
        show_sync_status()
"""
