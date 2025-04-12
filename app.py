import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib

# 設置 Matplotlib 支持中文，並使用合適的字體
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 嘗試使用微軟正黑體
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 網頁設定
st.set_page_config(page_title="頻域紋理分類工具", layout="wide")
st.title("📊 圖片頻域紋理分類工具")
st.markdown("上傳圖片，系統將根據頻率統計特徵自動分類。")

uploaded_files = st.file_uploader(
    "📁 請上傳一組紋理圖片（至少 2 張）", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

# 頻域特徵萃取函式
def extract_frequency_features(image_array):
    img = cv2.resize(image_array, (256, 256))  # 將圖像調整為 256x256 大小
    f = np.fft.fft2(img)  # 計算圖像的 2D 傅里葉轉換
    fshift = np.fft.fftshift(f)  # 移動零頻率分量到圖像中心
    magnitude = np.abs(fshift)  # 計算頻譜的幅度
    flat_mag = magnitude.flatten()  # 將頻譜的幅度展平為一維數組

    # 計算頻域特徵
    mean_val = np.mean(flat_mag)  # 平均值
    std_val = np.std(flat_mag)  # 標準差
    skew_val = skew(flat_mag)  # 偏度
    kurt_val = kurtosis(flat_mag)  # 峰度
    high_freq_ratio = np.sum(flat_mag[flat_mag > mean_val]) / np.sum(flat_mag)  # 高頻率比率

    return [mean_val, std_val, skew_val, kurt_val, high_freq_ratio], magnitude

# 分析流程
if uploaded_files and len(uploaded_files) >= 2:
    st.success("✅ 開始進行分析與分類...")

    features = []  # 儲存每張圖像的特徵
    images = []  # 儲存圖像
    magnitudes = []  # 儲存頻譜
    filenames = []  # 儲存文件名

    for file in uploaded_files:
        image = Image.open(file).convert("L")  # 將圖片轉為灰階
        image_np = np.array(image)  # 轉為數組
        filenames.append(file.name)  # 儲存文件名

        feat, mag = extract_frequency_features(image_np)  # 提取頻域特徵
        features.append(feat)  # 儲存特徵
        images.append(image_np)  # 儲存圖像
        magnitudes.append(mag)  # 儲存頻譜

    # 自動調整群數
    num_clusters = min(len(features), 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)  # K-means 分類

    # 定義分類對應名稱
    categories = {0: "隨機", 1: "規則", 2: "不規則"}  # 定義分類對應的名稱

    # PCA 降維視覺化
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)  # 進行PCA降維

    st.subheader("📈 Classification Visualization (PCA Dimensionality Reduction)")
    fig, ax = plt.subplots(figsize=(10, 8))  # 增加圖表大小以確保顯示不擁擠
    for i in range(len(pca_features)):
        ax.scatter(pca_features[i, 0], pca_features[i, 1], c=f'C{labels[i]}', label=categories[labels[i]] if i == 0 else "")
        ax.text(pca_features[i, 0] + 0.01, pca_features[i, 1], filenames[i], fontsize=8, ha='left', va='bottom')
    ax.set_xlabel("PC1")  # X軸標籤
    ax.set_ylabel("PC2")  # Y軸標籤
    ax.set_title("Frequency Domain Statistical Features - PCA Visualization")  # 圖表標題
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)  # 顯示分類圖例

    # 顯示圖表
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

    # 顯示每張圖的分析結果
    st.subheader("🖼️ 原圖與分類結果")
    for i in range(len(images)):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(images[i], caption=f"{filenames[i]}", use_container_width=True)  # 使用use_container_width
        with col2:
            st.markdown(f"**分類結果：{categories[labels[i]]}**")  # 顯示分類名稱
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
