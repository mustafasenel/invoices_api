from paddleocr import PaddleOCR
import os

current_directory = os.getcwd()

model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, det_model_dir=f"{current_directory}/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer", rec_model_dir=f"{current_directory}/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer", cls_model_dir=f"{current_directory}/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer")

def get_text(img) -> str:
    result = model.ocr(img, cls=False)
    # OCR sonuçlarını ayıkla
    for iResult in result:
        for res in iResult:
            line_text = res[1][0]
    return line_text