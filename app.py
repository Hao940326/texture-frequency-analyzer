import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# 網頁設定
st.set_page_config(page_title="頻域紋理分類工具", layout="wide")
st.title("📊 圖片頻域紋理分類工具")
st.markdown("上傳圖片，系統將根據頻率統計特徵自動分類。")

uploaded_files = st.file_uploader(
    "📁 請上傳一組紋理圖片（至少 2 張）", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# 頻域特徵萃取函式
def extract_frequency_features(image_array):
    img = cv2.resize(image_array, (256, 256))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    flat_mag = magnitude.flatten()

    mean_val = np.mean(flat_mag)
    std_val = np.std(flat_mag)
    skew_val = skew(flat_mag)
    kurt_val = kurtosis(flat_mag)
    high_freq_ratio = np.sum(flat_mag[flat_mag > mean_val]) / np.sum(flat_mag)

    return [mean_val, std_val, skew_val, kurt_val, high_freq_ratio], magnitude

# 更新後的分類邏輯：規則、不規則、隨機
def assign_label(features):
    mean_val, std_val, skew_val, kurt_val, high_freq_ratio = features

    # 規則紋理的特徵條件：
    if high_freq_ratio > 0.2 and skew_val < 0.2 and kurt_val > 3:  
        return "規則"
    
    # 不規則紋理的特徵條件：
    elif kurt_val < 3 and skew_val > 0.5:
        return "不規則"

    # 隨機紋理的特徵條件：
    else:
        return "隨機"

# 分析流程
if uploaded_files and len(uploaded_files) >= 2:
    st.success("✅ 開始進行分析與分類...")

    features = []
    images = []
    magnitudes = []
    filenames = []

    for file in uploaded_files:
        image = Image.open(file).convert("L")
        image_np = np.array(image)
        filenames.append(file.name)

        feat, mag = extract_frequency_features(image_np)
        features.append(feat)
        images.append(image_np)
        magnitudes.append(mag)

    # 自動調整群數
    num_clusters = min(len(features), 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # 分配每張圖片的類別（規則、不規則、隨機）
    assigned_labels = [assign_label(feat) for feat in features]

    # PCA 降維視覺化
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    st.subheader("📈 分類視覺化（PCA 降維）")
    fig, ax = plt.subplots()
    for i in range(len(pca_features)):
        ax.scatter(pca_features[i, 0], pca_features[i, 1], c=f'C{labels[i]}')
        ax.text(pca_features[i, 0] + 0.01, pca_features[i, 1], filenames[i], fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("頻域統計特徵 - PCA 視覺化")
    ax.grid(True)
    st.pyplot(fig)

    # 顯示每張圖的分析結果
    st.subheader("🖼️ 原圖與分類結果")
    for i in range(len(images)):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(images[i], caption=f"{filenames[i]}", use_column_width=True)
        with col2:
            st.markdown(f"**分類結果：{assigned_labels[i]}**")
            st.markdown("**頻域統計特徵：**")
            st.json({
                "Mean": round(features[i][0], 2),
                "Std": round(features[i][1], 2),
                "Skew": round(features[i][2], 2),
                "Kurtosis": round(features[i][3], 2),
                "HighFreqRatio": round(features[i][4], 3),
            })

else:
    st.info("請上傳至少兩張圖來進行分類。")
