import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import hashlib
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
import json

# ===========================
# KONFIGURASI HALAMAN
# ===========================
st.set_page_config(
    page_title="PPIC Forecasting Pro",
    page_icon="📊",
    layout="wide"
)

# ===========================
# DATABASE CONFIGURATION
# ===========================

@st.cache_resource
def init_connection():
    """Initialize PostgreSQL connection"""
    try:
        # Ambil connection string dari Streamlit Secrets
        conn = psycopg2.connect(st.secrets["database"]["url"])
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

def create_tables():
    """Create tables if not exists"""
    conn = init_connection()
    cur = conn.cursor()
    
    # Table untuk users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(50) PRIMARY KEY,
            password VARCHAR(64) NOT NULL,
            email VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table untuk forecast history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_history (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) REFERENCES users(username) ON DELETE CASCADE,
            forecast_data JSONB NOT NULL,
            parameters JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()

# Initialize database
create_tables()

# ===========================
# DATABASE FUNCTIONS
# ===========================

def hash_password(password):
    """Hash password menggunakan SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user_db(username, password, email):
    """Register user baru ke database"""
    conn = init_connection()
    cur = conn.cursor()
    
    try:
        # Cek apakah username sudah ada
        cur.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return False, "Username sudah digunakan!"
        
        # Insert user baru
        cur.execute(
            "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
            (username, hash_password(password), email)
        )
        conn.commit()
        cur.close()
        return True, "Registrasi berhasil! Silakan login."
    except Exception as e:
        conn.rollback()
        cur.close()
        return False, f"Error: {str(e)}"

def login_user_db(username, password):
    """Login user dari database"""
    conn = init_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    cur.close()
    
    if not result:
        return False, "Username tidak ditemukan!"
    
    if result[0] == hash_password(password):
        return True, "Login berhasil!"
    else:
        return False, "Password salah!"

def save_forecast_to_db(username, forecast_data, params):
    """Simpan hasil forecast ke database"""
    conn = init_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "INSERT INTO forecast_history (username, forecast_data, parameters) VALUES (%s, %s, %s)",
            (username, Json(forecast_data), Json(params))
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        cur.close()
        st.error(f"Error saving forecast: {e}")
        return False

def get_user_history_db(username):
    """Ambil history forecast user dari database"""
    conn = init_connection()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT id, forecast_data, parameters, created_at FROM forecast_history WHERE username = %s ORDER BY created_at DESC",
        (username,)
    )
    results = cur.fetchall()
    cur.close()
    
    history = []
    for row in results:
        history.append({
            'id': row[0],
            'forecast_data': row[1],
            'params': row[2],
            'timestamp': row[3].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return history

def get_user_info_db(username):
    """Ambil info user dari database"""
    conn = init_connection()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT email, created_at FROM users WHERE username = %s",
        (username,)
    )
    result = cur.fetchone()
    cur.close()
    
    if result:
        return {
            'email': result[0],
            'created_at': result[1].strftime("%Y-%m-%d %H:%M:%S")
        }
    return None

def delete_user_history_db(username):
    """Hapus semua history user"""
    conn = init_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("DELETE FROM forecast_history WHERE username = %s", (username,))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        cur.close()
        return False

def get_user_stats_db(username):
    """Ambil statistik user"""
    conn = init_connection()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT COUNT(*) FROM forecast_history WHERE username = %s",
        (username,)
    )
    total = cur.fetchone()[0]
    
    cur.execute(
        "SELECT created_at FROM forecast_history WHERE username = %s ORDER BY created_at DESC LIMIT 1",
        (username,)
    )
    last = cur.fetchone()
    last_activity = last[0].strftime("%Y-%m-%d %H:%M:%S") if last else "Belum ada"
    
    cur.close()
    
    return {
        'total_forecasts': total,
        'last_activity': last_activity
    }

# ===========================
# SESSION STATE MANAGEMENT
# ===========================

def init_session_state():
    """Initialize session state"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None

init_session_state()

# ===========================
# NEURAL NETWORK FUNCTIONS
# ===========================

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.hidden1_weights = np.random.uniform(-1, 1, (input_size, hidden1_size))
        self.hidden1_bias = np.random.uniform(-1, 1, (1, hidden1_size))
        self.hidden2_weights = np.random.uniform(-1, 1, (hidden1_size, hidden2_size))
        self.hidden2_bias = np.random.uniform(-1, 1, (1, hidden2_size))
        self.output_weights = np.random.uniform(-1, 1, (hidden2_size, output_size))
        self.output_bias = np.random.uniform(-1, 1, (1, output_size))
    
    def forward(self, X):
        self.hidden1_input = np.dot(X, self.hidden1_weights) + self.hidden1_bias
        self.hidden1_output = sigmoid(self.hidden1_input)
        self.hidden2_input = np.dot(self.hidden1_output, self.hidden2_weights) + self.hidden2_bias
        self.hidden2_output = sigmoid(self.hidden2_input)
        self.final_input = np.dot(self.hidden2_output, self.output_weights) + self.output_bias
        self.final_output = sigmoid(self.final_input)
        return self.final_output
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)
        self.hidden2_error = np.dot(self.output_delta, self.output_weights.T)
        self.hidden2_delta = self.hidden2_error * sigmoid_derivative(self.hidden2_output)
        self.hidden1_error = np.dot(self.hidden2_delta, self.hidden2_weights.T)
        self.hidden1_delta = self.hidden1_error * sigmoid_derivative(self.hidden1_output)
        self.output_weights += np.dot(self.hidden2_output.T, self.output_delta) * self.learning_rate
        self.output_bias += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.hidden2_weights += np.dot(self.hidden1_output.T, self.hidden2_delta) * self.learning_rate
        self.hidden2_bias += np.sum(self.hidden2_delta, axis=0, keepdims=True) * self.learning_rate
        self.hidden1_weights += np.dot(X.T, self.hidden1_delta) * self.learning_rate
        self.hidden1_bias += np.sum(self.hidden1_delta, axis=0, keepdims=True) * self.learning_rate
        return np.mean(self.output_error ** 2)

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def forecast_future(model, last_known_input, steps, data_min, data_max):
    forecast_results_norm = []
    current_input = last_known_input
    
    for _ in range(steps):
        next_pred_norm = model.forward(current_input)
        forecast_results_norm.append(next_pred_norm[0][0])
        current_input = next_pred_norm
    
    forecast_results_denorm = np.array(forecast_results_norm) * (data_max - data_min) + data_min
    return forecast_results_denorm

# ===========================
# LOGIN/REGISTER PAGE
# ===========================

if not st.session_state.logged_in:
    st.title("🔐 PPIC Forecasting System - Login")
    st.markdown("### Sistem Forecasting dengan PostgreSQL Database")
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login ke Akun Anda")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("🚀 Login", type="primary", use_container_width=True):
            if login_username and login_password:
                success, message = login_user_db(login_username, login_password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Mohon isi username dan password!")
    
    with tab2:
        st.subheader("Buat Akun Baru")
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_password2 = st.text_input("Konfirmasi Password", type="password", key="reg_pass2")
        
        if st.button("📝 Register", type="primary", use_container_width=True):
            if reg_username and reg_email and reg_password and reg_password2:
                if reg_password != reg_password2:
                    st.error("Password tidak cocok!")
                elif len(reg_password) < 6:
                    st.error("Password minimal 6 karakter!")
                else:
                    success, message = register_user_db(reg_username, reg_password, reg_email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Mohon isi semua field!")
    
    st.markdown("---")
    st.success("✅ **Database:** PostgreSQL (Production Ready)")
    st.info("💡 Data tersimpan permanen di cloud database")

else:
    # ===========================
    # MAIN APP (SETELAH LOGIN)
    # ===========================
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("📊 PPIC Forecasting System Pro")
    with col2:
        st.metric("👤 User", st.session_state.current_user)
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.rerun()
    
    st.markdown("---")
    
    page = st.sidebar.radio(
        "📑 Menu Navigasi:",
        ["🏠 Forecasting", "📜 History", "👤 Profile"]
    )
    
    if page == "🏠 Forecasting":
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Pengaturan")
        
        input_method = st.sidebar.radio(
            "Metode Input Data:",
            ["Input Manual", "Upload CSV"]
        )
        
        demand = None
        
        if input_method == "Input Manual":
            st.sidebar.subheader("Input Data Demand")
            demand_input = st.sidebar.text_area(
                "Masukkan data demand (pisahkan dengan koma):",
                value="3000, 0, 5000, 3500, 0, 6000, 3000, 0, 0, 2000, 8000, 3000",
                height=100
            )
            try:
                demand = np.array([float(x.strip()) for x in demand_input.split(',')])
            except:
                st.sidebar.error("❌ Format data tidak valid!")
        else:
            uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'demand' in df.columns:
                        demand = df['demand'].values
                    else:
                        demand = df.iloc[:, 0].values
                    st.sidebar.success(f"✅ Data berhasil diupload ({len(demand)} data)")
                except:
                    st.sidebar.error("❌ Gagal membaca file CSV!")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Parameter Model")
        
        window_size = st.sidebar.slider("Window Size (Smoothing):", 2, 10, 6)
        jumlah_bulan_forecast = st.sidebar.slider("Jumlah Bulan Forecast:", 1, 24, 12)
        
        with st.sidebar.expander("⚡ Advanced Settings"):
            hidden1_size = st.selectbox("Hidden Layer 1:", [4, 6, 8], index=1)
            hidden2_size = st.selectbox("Hidden Layer 2:", [4, 6], index=0)
            learning_rate = st.selectbox("Learning Rate:", [0.01, 0.05], index=1)
            epochs = st.selectbox("Epochs:", [5000, 10000], index=1)
        
        start_button = st.sidebar.button("🚀 Mulai Forecasting", type="primary", use_container_width=True)
        
        if demand is not None and len(demand) > 0:
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Data & Visualisasi", "🤖 Model Training", "🔮 Hasil Forecasting", "📥 Download"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Data Demand Historis")
                    df_display = pd.DataFrame({
                        'Bulan': range(1, len(demand) + 1),
                        'Demand': demand.astype(int)
                    })
                    st.dataframe(df_display, use_container_width=True, height=300)
                    
                    st.metric("Total Data Points", len(demand))
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Mean", f"{demand.mean():.0f}")
                    col_b.metric("Max", f"{demand.max():.0f}")
                    col_c.metric("Min", f"{demand.min():.0f}")
                
                with col2:
                    st.subheader("📉 Visualisasi Data")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(len(demand)), demand, marker='o', linestyle='-', linewidth=2, markersize=6)
                    ax.set_xlabel('Bulan', fontsize=12)
                    ax.set_ylabel('Demand', fontsize=12)
                    ax.set_title('Data Demand Historis', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            if start_button:
                with st.spinner('🔄 Sedang memproses data dan training model...'):
                    
                    demand_smooth = np.convolve(demand, np.ones(window_size)/window_size, mode='valid')
                    data_min, data_max = demand_smooth.min(), demand_smooth.max()
                    demand_norm = (demand_smooth - data_min) / (data_max - data_min)
                    
                    X = demand_norm[:-1].reshape(-1, 1)
                    y = demand_norm[1:].reshape(-1, 1)
                    
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    with tab2:
                        st.subheader("🤖 Proses Training Model")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        nn = NeuralNetwork(X_train.shape[1], hidden1_size, hidden2_size, 1, learning_rate)
                        
                        for epoch in range(epochs):
                            output = nn.forward(X_train)
                            nn.backward(X_train, y_train, output)
                            
                            if epoch % (epochs // 10) == 0:
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Training... Epoch {epoch}/{epochs}")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Training selesai!")
                        
                        y_pred_test = nn.forward(X_test)
                        mse_test = np.mean((y_test - y_pred_test)**2)
                        mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
                        
                        st.success("✅ Model berhasil ditraining!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Architecture", f"1-{hidden1_size}-{hidden2_size}-1")
                        col2.metric("Learning Rate", learning_rate)
                        col3.metric("MSE (Test)", f"{mse_test:.6f}")
                        col4.metric("MAPE (Test)", f"{mape_test:.2f}%")
                        
                        st.subheader("📊 Evaluasi Model: Actual vs Predicted")
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(range(len(y_test)), y_test, 'o-', label='Actual', linewidth=2)
                        ax.plot(range(len(y_test)), y_pred_test, 's-', label='Predicted', linewidth=2)
                        ax.set_xlabel('Data Point', fontsize=12)
                        ax.set_ylabel('Normalized Value', fontsize=12)
                        ax.set_title('Model Performance on Test Data', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=12)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with tab3:
                        st.subheader("🔮 Hasil Forecasting")
                        
                        last_input_norm = demand_norm[-1].reshape(1, 1)
                        hasil_forecast = forecast_future(
                            model=nn,
                            last_known_input=last_input_norm,
                            steps=jumlah_bulan_forecast,
                            data_min=data_min,
                            data_max=data_max
                        )
                        
                        df_forecast = pd.DataFrame({
                            'Bulan': [f"+{i+1}" for i in range(jumlah_bulan_forecast)],
                            'Prediksi Demand': [int(val) for val in hasil_forecast]
                        })
                        
                        forecast_params = {
                            'window_size': window_size,
                            'hidden1': hidden1_size,
                            'hidden2': hidden2_size,
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'mse': float(mse_test),
                            'mape': float(mape_test)
                        }
                        
                        if save_forecast_to_db(st.session_state.current_user, df_forecast.to_dict(), forecast_params):
                            st.success("💾 Hasil forecast telah disimpan ke PostgreSQL database!")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(df_forecast, use_container_width=True, height=400)
                            
                            st.markdown("**📊 Statistik Forecast:**")
                            st.metric("Rata-rata", f"{hasil_forecast.mean():.0f}")
                            st.metric("Maksimum", f"{hasil_forecast.max():.0f}")
                            st.metric("Minimum", f"{hasil_forecast.min():.0f}")
                        
                        with col2:
                            demand_smooth_denorm = demand_norm * (data_max - data_min) + data_min
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(range(len(demand_smooth_denorm)), demand_smooth_denorm, 
                                   'o-', label='Data Historis', linewidth=2, markersize=6)
                            ax.plot(range(len(demand_smooth_denorm), len(demand_smooth_denorm) + jumlah_bulan_forecast), 
                                   hasil_forecast, 's--', label=f'Forecast {jumlah_bulan_forecast} Bulan', 
                                   linewidth=2, markersize=6, color='red')
                            ax.axvline(x=len(demand_smooth_denorm)-1, color='gray', linestyle=':', 
                                      linewidth=2, alpha=0.7, label='Batas Historis')
                            ax.set_xlabel('Periode', fontsize=12)
                            ax.set_ylabel('Demand', fontsize=12)
                            ax.set_title('Peramalan Demand PPIC', fontsize=14, fontweight='bold')
                            ax.legend(fontsize=11)
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    
                    with tab4:
                        st.subheader("📥 Download Hasil")
                        
                        csv_buffer = io.StringIO()
                        df_forecast.to_csv(csv_buffer, index=False)
                        csv_str = csv_buffer.getvalue()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="📄 Download Hasil Forecast (CSV)",
                                data=csv_str,
                                file_name=f"forecast_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            report = f"""LAPORAN FORECASTING PPIC
========================
User: {st.session_state.current_user}
Tanggal: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Parameter Model:
- Architecture: 1-{hidden1_size}-{hidden2_size}-1
- Learning Rate: {learning_rate}
- Epochs: {epochs}
- Window Size: {window_size}

Performa Model:
- MSE (Test): {mse_test:.6f}
- MAPE (Test): {mape_test:.2f}%

Hasil Forecasting {jumlah_bulan_forecast} Bulan:
{df_forecast.to_string()}

Statistik Forecast:
- Rata-rata: {hasil_forecast.mean():.0f}
- Maksimum: {hasil_forecast.max():.0f}
- Minimum: {hasil_forecast.min():.0f}
"""
                            
                            st.download_button(
                                label="📋 Download Report Lengkap (TXT)",
                                data=report,
                                file_name=f"report_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
        
        else:
            st.info("👈 Silakan input data demand melalui sidebar untuk memulai forecasting")
    
    elif page == "📜 History":
        st.header("📜 History Forecasting")
        
        history = get_user_history_db(st.session_state.current_user)
        
        if len(history) == 0:
            st.info("Belum ada history forecasting. Mulai forecasting untuk menyimpan history!")
        else:
            st.success(f"📊 Total {len(history)} forecasting telah dilakukan")
            
            for idx, entry in enumerate(history):
                with st.expander(f"🔍 Forecast #{idx + 1} - {entry['timestamp']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Hasil Prediksi")
                        df_hist = pd.DataFrame(entry['forecast_data'])
                        st.dataframe(df_hist, use_container_width=True)
                    
                    with col2:
                        st.subheader("Parameter Model")
                        params = entry['params']
                        st.write(f"**Window Size:** {params['window_size']}")
                        st.write(f"**Architecture:** 1-{params['hidden1']}-{params['hidden2']}-1")
                        st.write(f"**Learning Rate:** {params['learning_rate']}")
                        st.write(f"**Epochs:** {params['epochs']}")
                        st.write(f"**MSE:** {params['mse']:.6f}")
                        st.write(f"**MAPE:** {params['mape']:.2f}%")
                    
                    csv_buffer = io.StringIO()
                    df_hist.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Hasil Ini (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"history_{entry['id']}_{entry['timestamp'].replace(' ', '_').replace(':', '-')}.csv",
                        mime="text/csv",
                        key=f"download_{entry['id']}"
                    )
    
    elif page == "👤 Profile":
        st.header("👤 Profile User")
        
        user_info = get_user_info_db(st.session_state.current_user)
        stats = get_user_stats_db(st.session_state.current_user)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informasi Akun")
            st.write(f"**Username:** {st.session_state.current_user}")
            st.write(f"**Email:** {user_info['email']}")
            st.write(f"**Member Since:** {user_info['created_at']}")
            st.write(f"**Total Forecasts:** {stats['total_forecasts']}")
        
        with col2:
            st.subheader("Statistik Penggunaan")
            st.metric("Total Forecasting", stats['total_forecasts'])
            st.metric("Last Activity", stats['last_activity'])
        
        st.markdown("---")
        
        st.subheader("⚙️ Pengaturan")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🗑️ Hapus Semua History", type="secondary", use_container_width=True):
                if delete_user_history_db(st.session_state.current_user):
                    st.success("History berhasil dihapus!")
                    st.rerun()
                else:
                    st.error("Gagal menghapus history!")
        
        with col_b:
            st.info("💾 Data tersimpan permanen di PostgreSQL")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>PPIC Forecasting System Pro | PostgreSQL Database</div>",
    unsafe_allow_html=True
)