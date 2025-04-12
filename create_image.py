import numpy as np
import cv2
import os

# 建立紋理輸出資料夾
output_folder = 'generated_textures'
os.makedirs(output_folder, exist_ok=True)

# 規則紋理：棋盤格
def generate_regular_texture(size=256, square_size=32):
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, square_size):
        for x in range(0, size, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = 255
    return img

# 不規則紋理：模糊後的噪聲
def generate_irregular_texture(size=256):
    noise = np.random.rand(size, size).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (15, 15), 5)
    noise = (255 * noise).astype(np.uint8)
    return noise

# 隨機紋理：純噪聲
def generate_random_texture(size=256):
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)

# 生成紋理
regular = generate_regular_texture()
irregular = generate_irregular_texture()
random = generate_random_texture()

# 儲存圖片
cv2.imwrite(os.path.join(output_folder, 'regular_texture.png'), regular)
cv2.imwrite(os.path.join(output_folder, 'irregular_texture.png'), irregular)
cv2.imwrite(os.path.join(output_folder, 'random_texture.png'), random)

print("✅ 紋理圖像已儲存到資料夾：generated_textures")
