import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. KONFIGURASI FOLDER
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"
MASK_DIR = OUTPUT_DIR / "masks"

# Membuat folder output jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "hasil_monitoring.csv"
GRAPH_FILE = OUTPUT_DIR / "grafik_persentase_hijau.png"

# ==========================================
# 2. PARAMETER SEGMENTASI HSV
# ==========================================
lower_green = np.array([25, 35, 30], dtype=np.uint8)
upper_green = np.array([95, 255, 255], dtype=np.uint8)
kernel = np.ones((5, 5), dtype=np.uint8)
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================
# 3. MENCARI GAMBAR INPUT
# ==========================================
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Folder gambar tidak ditemukan: {INPUT_DIR}")

image_files = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
])

if not image_files:
    raise FileNotFoundError(f"Tidak ada gambar JPG, JPEG, PNG, atau BMP di {INPUT_DIR}")

data = []

# ==========================================
# 4. MEMPROSES SETIAP GAMBAR
# ==========================================
for image_path in image_files:
    print(f"\nMemproses: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Peringatan: gambar gagal dibaca: {image_path.name}")
        continue

    # Konversi BGR ke RGB untuk tampilan Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Konversi BGR ke HSV untuk segmentasi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Segmentasi berdasarkan rentang HSV
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Membersihkan noise dan lubang kecil (Operasi Morfologi)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    # Menghitung piksel
    total_pixels = mask_clean.size
    green_pixels = cv2.countNonZero(mask_clean)
    percentage = (green_pixels / total_pixels * 100) if total_pixels > 0 else 0

    # Menghasilkan citra hasil segmentasi
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)
    
    # Menyimpan mask
    mask_path = MASK_DIR / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_clean)
    
    # Mengambil ID tanaman dari bagian awal nama file (misal: P001_...)
    plant_id = image_path.stem.split("_")[0]
    
    data.append({
        "waktu_pemrosesan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id_tanaman": plant_id,
        "nama_file": image_path.name,
        "lebar_piksel": img.shape[1],
        "tinggi_piksel": img.shape[0],
        "total_piksel": total_pixels,
        "piksel_hijau": green_pixels,
        "persentase_hijau": round(percentage, 2),
        "hsv_bawah": str(lower_green.tolist()),
        "hsv_atas": str(upper_green.tolist()),
        "file_mask": str(mask_path.relative_to(BASE_DIR))
    })

    print(f"  Total piksel    : {total_pixels:,}")
    print(f"  Piksel hijau    : {green_pixels:,}")
    print(f"  Persentase      : {percentage:.2f}%")

    # Menampilkan gambar asli, mask, dan hasil segmentasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Gambar Asli\n{image_path.name}")
    axes[0].axis("off")

    axes[1].imshow(mask_clean, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Mask Segmentasi Hijau")
    axes[1].axis("off")

    axes[2].imshow(segmented)
    axes[2].set_title("Hasil Segmentasi")
    axes[2].axis("off")

    fig.tight_layout()
    plt.show()
    plt.close(fig)

# ==========================================
# 5. MENYIMPAN HASIL KE CSV DAN MEMBUAT GRAFIK
# ==========================================
if data:
    df_new = pd.DataFrame(data)
    
    # Menggabungkan dengan data CSV lama jika sudah ada
    if CSV_FILE.exists():
        df_old = pd.read_csv(CSV_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # Simpan ke CSV
    df_all.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print("\nMonitoring selesai.")
    print(f"Jumlah gambar berhasil diproses: {len(data)}")
    print(f"Data tersimpan di: {CSV_FILE}")

    print("\nRingkasan hasil pemrosesan saat ini:")
    print(df_new[["id_tanaman", "nama_file", "piksel_hijau", "persentase_hijau"]].to_string(index=False))

    # Membuat grafik persentase piksel hijau
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_new["nama_file"], df_new["persentase_hijau"], color='green', alpha=0.7)
    ax.set_title("Persentase Piksel Tersegmentasi Hijau")
    ax.set_xlabel("Nama File")
    ax.set_ylabel("Persentase (%)")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(GRAPH_FILE, dpi=300, bbox_inches="tight")
    print(f"Grafik tersimpan di: {GRAPH_FILE}")

    plt.show()
    plt.close(fig)
else:
    print("Tidak ada gambar yang berhasil diproses.")