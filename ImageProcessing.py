from PIL import Image, ImageEnhance
import numpy as np
from scipy import ndimage
from io import BytesIO
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt



def region_growing1(img, seeds, threshold=30):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    if isinstance(seeds, tuple):
        seeds = [seeds]

    for seed in seeds:
        mask[seed[0], seed[1]] = 255
        seed_value = img[seed[0], seed[1]]

        queue = [seed]
        while queue:
            x, y = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                    if abs(int(img[nx, ny]) - int(seed_value)) < threshold:
                        mask[nx, ny] = 255
                        queue.append((nx, ny))

    return mask


def display_rgb_channels(image):
    image_array = np.array(image)
    r, g, b = cv2.split(image_array)

    def create_single_channel_image(channel, color):
        blank_channel = np.zeros_like(channel)
        if color == 'red':
            return cv2.merge((channel, blank_channel, blank_channel))
        elif color == 'green':
            return cv2.merge((blank_channel, channel, blank_channel))
        elif color == 'blue':
            return cv2.merge((blank_channel, blank_channel, channel))

    red_image_array = create_single_channel_image(r, 'red')
    green_image_array = create_single_channel_image(g, 'green')
    blue_image_array = create_single_channel_image(b, 'blue')

    red_image = Image.fromarray(red_image_array)
    green_image = Image.fromarray(green_image_array)
    blue_image = Image.fromarray(blue_image_array)

    return red_image, green_image, blue_image


def plot_grayscale_histogram_as_image(image):
    # Görüntü numpy dizisine dönüştürülür
    image_array = np.array(image)

    # Gri tonlama görüntü olup olmadığı kontrol edilir
    if len(image_array.shape) != 2:
        # Eğer gri tonlama değilse, gri tonlamaya dönüştürülür
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Histogram çizimi
    plt.figure()
    plt.hist(image_array.ravel(), 256, [0, 256], color='black')
    plt.title('Histogram for Grayscale Image')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')

    # Histogram görüntüsünün kaydedilmesi
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()

    # Bellekten görüntü okunur
    pil_image = Image.open(buf)
    open_cv_image = np.array(pil_image)
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)



def plot_histogram_as_image(image):
    image_array = np.array(image)
    plt.figure()
    if len(image_array.shape) == 2:
        plt.hist(image_array.ravel(), 256, [0, 256])
        plt.title('Histogram for Grayscale Image')
    elif len(image_array.shape) == 3:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram for Color Image')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()

    # Read buffer as an image
    pil_image = Image.open(buf)
    open_cv_image = np.array(pil_image)
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def histogram_equalization(image):
    image_array = np.array(image)
    r, g, b = cv2.split(image_array)
    def equalize_channel(channel):
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        equalized_channel = cdf[channel]
        return equalized_channel
    r_equalized = equalize_channel(r)
    g_equalized = equalize_channel(g)
    b_equalized = equalize_channel(b)

    equalized_image_array = cv2.merge((r_equalized, g_equalized, b_equalized))
    equalized_image = Image.fromarray(equalized_image_array)
    return equalized_image


def grayscale_conversion(image):
    img_array = np.array(image, dtype=np.float32)
    height, width, _ = img_array.shape
    grayscale_array = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            r, g, b = img_array[y, x]
            grayscale_value = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_array[y, x] = grayscale_value

    grayscale_array = np.clip(grayscale_array, 0, 255).astype(np.uint8)
    grayscale_image = Image.fromarray(grayscale_array, mode='L')

    return grayscale_image


def binary_conversion(image, threshold=128):
    img_array = np.array(image, dtype=np.float32)
    height, width, _ = img_array.shape
    grayscale_array = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            r, g, b = img_array[y, x]
            grayscale_value = 0.299 * r + 0.587 * g + 0.114 * b
            grayscale_array[y, x] = grayscale_value

    binary_array = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            binary_array[y, x] = 255 if grayscale_array[y, x] > threshold else 0

    binary_image = Image.fromarray(binary_array, mode='L')

    return binary_image


def adjust_brightness(image, brightness):
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                img_array[y, x, c] = img_array[y, x, c] * brightness

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    adjusted_image = Image.fromarray(img_array)

    return adjusted_image

def zoom_image(image, zoom_factor):
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    if zoom_factor > 1:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        image = image.crop((left, top, right, bottom))
    else:
        new_image = Image.new("RGB", (width, height))
        new_image.paste(image, ((width - new_width) // 2, (height - new_height) // 2))
        image = new_image

    return image

def resize_image(image, width, height):
    resized_image = image.resize((width, height))
    return resized_image


def apply_morphological_operation(image, operation="opening", size=3):
    img_array = np.array(image)
    structuring_element = np.ones((size, size), dtype=bool)
    if operation == "opening":
        result_image = ndimage.binary_opening(img_array, structure=structuring_element)
    elif operation == "closing":
        result_image = ndimage.binary_closing(img_array, structure=structuring_element)
    elif operation == "erosion":
        result_image = ndimage.binary_erosion(img_array, structure=structuring_element)
    elif operation == "dilation":
        result_image = ndimage.binary_dilation(img_array, structure=structuring_element)
    else:
        raise ValueError("Invalid morphological operation. Please choose from 'opening', 'closing', 'erosion', or 'dilation'.")

    result_image = Image.fromarray(result_image.astype(np.uint8) * 255)
    return result_image


def adjust_contrast(image, contrast):
    image_array = np.array(image, dtype=np.float32)
    mean = np.mean(image_array)
    height, width, channels = image_array.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                image_array[y, x, c] = mean + contrast * (image_array[y, x, c] - mean)

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    adjusted_image = Image.fromarray(image_array)

    return adjusted_image


def region_growing(image, seeds, threshold=10):
    h, w = image.shape[:2]
    segmented = np.zeros_like(image)
    label = 1

    for seed in seeds:
        stack = [seed]
        while stack:
            x, y = stack.pop()
            if segmented[y, x] == 0:  # Check if the pixel is not already labeled
                segmented[y, x] = label
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and segmented[ny, nx] == 0:
                        if abs(int(image[ny, nx]) - int(image[y, x])) < threshold:
                            stack.append((nx, ny))
        label += 1

    segmented = ((segmented - segmented.min()) / (segmented.max() - segmented.min()) * 255).astype(np.uint8)
    return Image.fromarray(segmented)


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    features, hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, visualize=True)

    hog_image_rescaled = hog_image.astype(np.uint8)
    hog_image_rgb = cv2.cvtColor(hog_image_rescaled, cv2.COLOR_GRAY2RGB)
    hog_features_image = Image.fromarray(hog_image_rgb)
    return hog_features_image


def canny_edge_detection(image, low_threshold, high_threshold):
    image_array = np.array(image.convert("L"))
    edges = cv2.Canny(image_array, low_threshold, high_threshold)
    return Image.fromarray(edges)

def sobel_edge_detection(image, ksize):
    image_array = np.array(image.convert("L"))
    sobelx = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)
    return Image.fromarray(sobel)


def shift_image(image, shift_x, shift_y):
    image_array = np.array(image)
    shifted_image = np.roll(image_array, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)
    return Image.fromarray(shifted_image)

def flip_image(image, direction):
    if direction == "Horizontal":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "Vertical":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    img_array = np.array(image)
    blurred_image = cv2.GaussianBlur(img_array, kernel_size, sigma_x)
    blurred_image_pil = Image.fromarray(blurred_image)
    return blurred_image_pil



def rotate_image(image, angle):
    return image.rotate(angle, expand=True)


