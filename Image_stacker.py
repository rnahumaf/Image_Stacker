import cv2
import rawpy
import imageio
import numpy as np
from PIL import Image
from tkinter import filedialog, Tk

def sharpen_image(image, alpha=1.5, beta=-0.5):
    """
    Aplica o filtro de nitidez à imagem usando Unsharp Masking.
    
    Parâmetros:
    - image: Imagem de entrada.
    - alpha: Fator de contraste. Um valor de 1.0 não tem efeito.
    - beta: Fator de brilho. Um valor de 0.0 não tem efeito.
    
    Retorna:
    - Imagem com nitidez aumentada.
    """
    
    # Suavizar a imagem usando filtro Gaussiano
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Calcula a máscara de nitidez
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    
    return sharpened

def align_ecc_subpixel(base_image, image, resize_factor=1):
    # Aumentar a resolução das imagens
    base_resized = cv2.resize(base_image, (0, 0), fx=resize_factor, fy=resize_factor)
    image_resized = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

    # Alinhamento usando ECC na imagem redimensionada
    aligned_resized = align_ecc(base_resized, image_resized)

    # Redimensionar a imagem alinhada de volta à resolução original
    h, w = base_image.shape[:2]
    aligned_image = cv2.resize(aligned_resized, (w, h))

    return aligned_image

def align_ecc(base_image, image):
    # Converter imagens para escala de cinza
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Garantir que ambas as imagens tenham o mesmo tamanho
    base_gray = cv2.resize(base_gray, (image_gray.shape[1], image_gray.shape[0]))

    # Estimar a transformação afim usando a maximização ECC
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    
    try:
        _, warp_matrix = cv2.findTransformECC(base_gray, image_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
    except cv2.error as err:
        print("Erro durante a maximização ECC:", err)
        return image

    h, w = base_image.shape[:2]
    aligned_image = cv2.warpAffine(image, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned_image


def read_image(image_path):
    ext = image_path.split('.')[-1].lower()
    if ext in ["jpg", "jpeg"]:
        return cv2.imread(image_path)
    elif ext == "heic":
        return cv2.cvtColor(imageio.imread(image_path), cv2.COLOR_RGB2BGR)
    elif ext in ["raw", "dng"]:
        with rawpy.imread(image_path) as raw:
            return cv2.cvtColor(raw.postprocess(), cv2.COLOR_RGB2BGR)
    elif ext == "tiff":
        return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    raise ValueError(f"Formato {ext} não suportado")

def select_images():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Selecione as imagens", 
        filetypes=[("Arquivos de imagem", "*.jpg;*.jpeg;*.heic;*.tiff;*.raw;*.dng")]
    )
    directory = file_paths[0].rpartition('/')[0] if file_paths else ""
    return [read_image(image_path) for image_path in file_paths], directory

def median_images(images):
    return np.median(images, axis=0).astype(np.uint8)

def crop_to_smallest(images):
    min_h, min_w = min(img.shape[:2] for img in images)
    return [img[:min_h, :min_w] for img in images]

def save_image(image, directory, filename):
    ext = filename.split('.')[-1].lower()
    save_path = f"{directory}/{filename}_processed"
    if ext in ["jpg", "jpeg"]:
        cv2.imwrite(save_path + ".jpg", image)
    elif ext == "heic":
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(save_path + ".heic", "HEIC")
    elif ext in ["raw", "dng", "tiff"]:
        cv2.imwrite(save_path + ".tiff", image)

def main():
    # Carrega as imagens e obtém o diretório
    print("Carregando imagens")
    images, directory = select_images()

    if not images:
        print("Nenhuma imagem selecionada. Encerrando.")
        return

    base_image = images[0]  # Definindo a primeira imagem como imagem base

    print("Alinhando imagens")
    total_images = len(images)
    aligned_images = [base_image]

    for idx, img in enumerate(images[1:], start=1):
        aligned_img = align_ecc_subpixel(base_image, img)
        aligned_images.append(aligned_img)

        # Imprimir progresso
        percentage_done = (idx / total_images) * 100
        print(f"Processado {idx} de {total_images} imagens ({percentage_done:.2f}% completo)")


    # Recorta as imagens
    print("Recortando imagens")
    cropped_images = crop_to_smallest(aligned_images)

    # Empilhar as imagens (Mediana)
    print("Empilhando imagens usando mediana...")
    stacked_image_median = median_images(cropped_images)

    # Aumentar a nitidez
    sharpened_image = sharpen_image(stacked_image_median)
    save_image(sharpened_image, directory, "imagem_empilhada_mediana.jpg")

if __name__ == "__main__":
    main()
