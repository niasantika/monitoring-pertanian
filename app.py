import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# Config Halaman Streamlit
st.set_page_config(
    page_title="Smart Plant Monitoring",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Smart Plant Monitoring - Segmentasi Warna Hijau")
st.write("Aplikasi segmentasi warna daun berbasis HSV menggunakan OpenCV dan Streamlit.")

# Sidebar untuk Pengaturan Parameter HSV
st.sidebar.header("⚙️ Parameter Segmentasi HSV")

h_min = st.sidebar.slider("Hue Min (H)", 0, 179, 25)
s_min = st.sidebar.slider("Saturation Min (S)", 0, 255, 35)
v_min = st.sidebar.slider("Value Min (V)", 0, 255, 30)

h_max = st.sidebar.slider("Hue Max (H)", 0, 179, 95)
s_max = st.sidebar.slider("Saturation Max (S)", 0, 255, 255)
v_max = st.sidebar.slider("Value Max (V)", 0, 255, 255)

lower_green = np.array([h_min, s_min, v_min], dtype=np.uint8)
upper_green = np.array([h_max, s_max, v_max], dtype=np.uint8)
kernel = np.ones((5, 5), dtype=np.uint8)

# Upload Gambar
st.subheader("📁 Unggah Foto Tanaman")
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa foto (JPG, PNG, BMP)...",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Berhasil mengunggah {len(uploaded_files)} gambar.")
    
    data_results = []
    
    st.subheader("🖼️ Hasil Proses Segmentasi")
    
    for uploaded_file in uploaded_files:
        # Membaca gambar dari buffer file upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error(f"Gagal membaca gambar: {uploaded_file.name}")
            continue

        # Konversi warna
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Segmentasi dan Operasi Morfologi
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
        
        # Menghitung piksel
        total_pixels = mask_clean.size
        green_pixels = cv2.countNonZero(mask_clean)
        percentage = (green_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        # Bitwise AND untuk hasil segmentasi
        segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)
        
        # Ambil ID Tanaman
        plant_id = uploaded_file.name.split("_")[0]
        
        data_results.append({
            "Waktu Pemrosesan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ID Tanaman": plant_id,
            "Nama File": uploaded_file.name,
            "Lebar Piksel": img.shape[1],
            "Tinggi Piksel": img.shape[0],
            "Total Piksel": total_pixels,
            "Piksel Hijau": green_pixels,
            "Persentase Hijau (%)": round(percentage, 2)
        })

        # Tampilan 3 Kolom untuk Gambar
        with st.expander(f"🔍 {uploaded_file.name} - Persentase: {percentage:.2f}%", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_rgb, caption="Gambar Asli", use_container_width=True)
            with col2:
                st.image(mask_clean, caption="Mask Segmentasi", use_container_width=True)
            with col3:
                st.image(segmented, caption="Hasil Segmentasi", use_container_width=True)

    # Menampilkan Rekap Data & Grafik
    if data_results:
        st.markdown("---")
        st.subheader("📊 Rekap Hasil Pemrosesan")
        df_results = pd.DataFrame(data_results)
        st.dataframe(df_results, use_container_width=True)
        
        # Tombol Download CSV
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Monitoring (CSV)",
            data=csv_data,
            file_name=f"hasil_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Visualisasi Grafik
        st.subheader("📈 Grafik Persentase Piksel Hijau")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df_results["Nama File"], df_results["Persentase Hijau (%)"], color='#2ecc71', alpha=0.85)
        ax.set_ylabel("Persentase (%)")
        ax.set_xlabel("Nama File")
        ax.set_title("Persentase Piksel Tersegmentasi Hijau")
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        
        st.pyplot(fig)

else:
    st.info("Silakan unggah foto pada area di atas untuk memulai pemrosesan.")