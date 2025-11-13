import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# ===========================
# KONFIGURASI HALAMAN
# ===========================
st.set_page_config(
    page_title="PPIC Forecasting System",
    page_icon="📊",
    layout="wide"
)

# ===========================
# FUNGSI-FUNGSI DARI CODE ASLI
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
# HEADER APLIKASI
# ===========================
st.title("📊 PPIC Forecasting System")
st.markdown("### Sistem Peramalan Permintaan dengan Neural Network")
st.markdown("---")

# ===========================
# SIDEBAR - INPUT DATA
# ===========================
st.sidebar.header("⚙️ Pengaturan")

# Pilihan input data
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

# ===========================
# PARAMETER MODEL
# ===========================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Parameter Model")

window_size = st.sidebar.slider("Window Size (Smoothing):", 2, 10, 6)
jumlah_bulan_forecast = st.sidebar.slider("Jumlah Bulan Forecast:", 1, 24, 12)

with st.sidebar.expander("⚡ Advanced Settings"):
    hidden1_size = st.selectbox("Hidden Layer 1:", [4, 6, 8], index=1)
    hidden2_size = st.selectbox("Hidden Layer 2:", [4, 6], index=0)
    learning_rate = st.selectbox("Learning Rate:", [0.01, 0.05], index=1)
    epochs = st.selectbox("Epochs:", [5000, 10000], index=1)

# ===========================
# TOMBOL MULAI FORECASTING
# ===========================
start_button = st.sidebar.button("🚀 Mulai Forecasting", type="primary", use_container_width=True)

# ===========================
# MAIN CONTENT
# ===========================

if demand is not None and len(demand) > 0:
    
    # Tab untuk organisasi konten
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
            
            # Statistik
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
            
            # Preprocessing
            demand_smooth = np.convolve(demand, np.ones(window_size)/window_size, mode='valid')
            data_min, data_max = demand_smooth.min(), demand_smooth.max()
            demand_norm = (demand_smooth - data_min) / (data_max - data_min)
            
            X = demand_norm[:-1].reshape(-1, 1)
            y = demand_norm[1:].reshape(-1, 1)
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Training
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
                
                # Evaluasi
                y_pred_test = nn.forward(X_test)
                mse_test = np.mean((y_test - y_pred_test)**2)
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
                
                st.success("✅ Model berhasil ditraining!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Architecture", f"1-{hidden1_size}-{hidden2_size}-1")
                col2.metric("Learning Rate", learning_rate)
                col3.metric("MSE (Test)", f"{mse_test:.6f}")
                col4.metric("MAPE (Test)", f"{mape_test:.2f}%")
                
                # Plot Actual vs Predicted
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
            
            # Forecasting
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
                
                # Tabel hasil
                df_forecast = pd.DataFrame({
                    'Bulan': [f"+{i+1}" for i in range(jumlah_bulan_forecast)],
                    'Prediksi Demand': [int(val) for val in hasil_forecast]
                })
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(df_forecast, use_container_width=True, height=400)
                    
                    # Statistik forecast
                    st.markdown("**📊 Statistik Forecast:**")
                    st.metric("Rata-rata", f"{hasil_forecast.mean():.0f}")
                    st.metric("Maksimum", f"{hasil_forecast.max():.0f}")
                    st.metric("Minimum", f"{hasil_forecast.min():.0f}")
                
                with col2:
                    # Plot forecast dengan historis
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
            
            # Download
            with tab4:
                st.subheader("📥 Download Hasil")
                
                # CSV Download
                csv_buffer = io.StringIO()
                df_forecast.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Hasil Forecast (CSV)",
                        data=csv_str,
                        file_name=f"forecast_result_{jumlah_bulan_forecast}_months.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Membuat report lengkap
                    report = f"""LAPORAN FORECASTING PPIC
========================

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
                        file_name="forecast_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.info("💡 **Tips:** Anda dapat menggunakan hasil ini untuk perencanaan produksi dan inventory management.")

else:
    st.info("👈 Silakan input data demand melalui sidebar untuk memulai forecasting")
    
    # Instruksi penggunaan
    st.markdown("""
    ### 📖 Cara Menggunakan:
    
    1. **Input Data**: Pilih metode input (manual atau upload CSV)
    2. **Set Parameter**: Atur parameter model di sidebar
    3. **Mulai Forecast**: Klik tombol "Mulai Forecasting"
    4. **Analisis Hasil**: Lihat hasil di berbagai tab
    5. **Download**: Unduh hasil untuk dokumentasi
    
    ---
    
    ### 📊 Format CSV:
    File CSV harus memiliki kolom bernama `demand` atau data di kolom pertama.
    
    Contoh:
    ```
    demand
    3000
    5000
    3500
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>PPIC Forecasting System | Powered by Neural Network</div>",
    unsafe_allow_html=True
)