import cv2
import pytesseract
import pdf2image
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

# Lista de palavras-chave a serem procuradas
keywords = ["60 (sessenta) dias", "bloqueio", "BACEN-JUD", "sequestro", "quitar a execução"]

def deskew(image):
    """Corrige a inclinação da imagem."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def extract_text_from_image(image):
    """Extrai texto da imagem."""
    text = pytesseract.image_to_string(image)
    return text

def process_page(page, keywords):
    """Processa uma página do PDF."""
    try:
        # Converte a página em um array NumPy
        page_arr = np.array(page)
        # Converte para tons de cinza
        page_arr_gray = cv2.cvtColor(page_arr, cv2.COLOR_BGR2GRAY)
        # Corrige a inclinação
        page_deskew = deskew(page_arr_gray)
        # Calcula a confiança (implementar a função get_conf)
        page_conf = get_conf(page_deskew)
        # Extrai dados da imagem
        d = pytesseract.image_to_data(page_deskew, output_type=pytesseract.Output.DICT)
        d_df = pd.DataFrame.from_dict(d)
        # Obtém o número do último bloco
        block_num = int(d_df.loc[d_df['level'] == 2, 'block_num'].max())
        # Remove cabeçalho e rodapé
        header_index = d_df[d_df['block_num'] == 1].index.values
        footer_index = d_df[d_df['block_num'] == block_num].index.values
        # Combina o texto do corpo da página
        text = ' '.join(d_df.loc[(d_df['level'] == 5) & (~d_df.index.isin(header_index) & ~d_df.index.isin(footer_index)), 'text'].values)

        # Extrai palavras-chave
        found_keywords = []
        for word in text.split():
            if word.lower() in keywords:
                found_keywords.append(word)

        return page_conf, text, found_keywords
    except Exception as e:
        if hasattr(e, 'message'):
            return -1, e.message
        else:
            return -1, str(e)

# Carrega o PDF
pdf_file = 'procsd.pdf'
pages = convert_from_path(pdf_file)

# Processa cada página
for page_num, page in enumerate(pages):
    page_conf, text, keywords_found = process_page(page, keywords)
    print(f"Página {page_num + 1}:")
    print(f"Palavras-chave encontradas: {keywords_found}")