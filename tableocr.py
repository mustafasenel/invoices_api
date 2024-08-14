from paddleocr import PaddleOCR
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

current_directory = os.getcwd()
# OCR modelini başlat
model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, det_model_dir=f"{current_directory}/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer", rec_model_dir=f"{current_directory}/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer", cls_model_dir=f"{current_directory}/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer")

# Resmi oku ve RGB formatına dönüştür
# image_path = "test_images/3.jpg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_list(img) -> list:
    result = model.ocr(img, cls=False)
    sorted_text = ocr_result_to_dataframe(result)
    df = pd.DataFrame(sorted_text)
    return sorted_text

def ocr_result_to_dataframe(result):
    all_texts = []
    all_coords = []

    # OCR sonuçlarını ayıkla
    for iResult in result:
        for res in iResult:
            line_text = res[1][0]
            box_coordinates = res[0]
            all_texts.append(line_text)
            all_coords.append(box_coordinates)

    # Her bir kutunun merkez y koordinatını hesapla
    centers_y = [np.mean([coord[1] for coord in box]) for box in all_coords]
    heights = [max(coord[1] for coord in box) - min(coord[1] for coord in box) for box in all_coords]

    # Ortalama yüksekliği hesapla ve eşik değeri olarak kullan
    avg_height = np.mean(heights)
    y_threshold = avg_height * 0.5  # Y ekseni farkı için eşik

    # DBSCAN kullanarak satırları gruplandır
    dbscan = DBSCAN(eps=y_threshold, min_samples=1)
    labels = dbscan.fit_predict(np.array(centers_y).reshape(-1, 1))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    grouped_texts = [[] for _ in range(n_clusters)]
    grouped_coords = [[] for _ in range(n_clusters)]
    for text, coords, label in zip(all_texts, all_coords, labels):
        if label != -1:
            grouped_texts[label].append(text)
            grouped_coords[label].append(coords)

    # Grupları y koordinatlarına göre sırala
    grouped_texts_coords = sorted(zip(grouped_texts, grouped_coords), key=lambda x: np.mean([np.mean([coord[1] for coord in box]) for box in x[1]]))

    sorted_texts = []
    for group_texts, group_coords in grouped_texts_coords:
        # Her grubu x koordinatına göre sırala (soldan sağa)
        sorted_group = sorted(zip(group_texts, group_coords), key=lambda x: x[1][0][0])
        if len(sorted_group) > 3:
            first_text, first_coord = sorted_group[0]  # Listenin ilk elemanını al
            if any(char.isalpha() for char in first_text):  # Harf kontrolü
                # Harf içeren bölümü ayır
                split_text = first_text.split()
                index = len(split_text[0])
                if index < len(first_text):  # İlk kelime sonrasında bir şey varsa
                    remaining_text = first_text[index:]
                    sorted_group[0] = (split_text[0], first_coord)  # İlk elemanı güncelle
                    # Yeni elemanı oluşturarak listeye ekle
                    sorted_group.insert(1, (remaining_text, first_coord))
            sorted_texts.append([text for text, _ in sorted_group])

    return sorted_texts

# # OCR sonuçlarını satırlara dönüştür
# grouped_text = ocr_result_to_dataframe(result)

# DataFrame'e dönüştür ve Excel dosyasına yaz
# df = pd.DataFrame(grouped_text)
# df.to_excel('ocr_results.xlsx', index=False, header=False)


# get_list(image)