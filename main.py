import streamlit as st
from PIL import Image
import cv2
from ImageProcessing import region_growing1, display_rgb_channels, plot_grayscale_histogram_as_image, grayscale_conversion, resize_image, binary_conversion, apply_morphological_operation, rotate_image, adjust_brightness, canny_edge_detection, sobel_edge_detection, zoom_image, shift_image, flip_image, adjust_contrast, histogram_equalization, region_growing,apply_gaussian_blur,extract_hog_features, plot_histogram_as_image
import numpy as np
import pandas as pd




clicked_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


def get_click_coordinates(image_array):
    global clicked_points
    clicked_points = []
    window_name = 'Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])

    # Convert the image to BGR format if it's RGB
    if image_array.shape[-1] == 3:
        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_array_bgr = image_array

    cv2.imshow(window_name, image_array_bgr)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
    return clicked_points


if 'result_images' not in st.session_state:
    st.session_state.result_images = []



def is_rgb_or_grayscale(image_array):
    if len(image_array.shape) == 2:
        return "Grayscale"
    elif len(image_array.shape) == 3 and image_array.shape[-1] == 3:
        return "RGB"
    elif len(image_array.shape) == 3 and image_array.shape[-1] == 4:
        return "RGBA"
    else:
        return "Unknown"



def main():
    st.title("Image Uploader or Downloader")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_type = is_rgb_or_grayscale(image_array)
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        fixed_size = (400, 400)
        image = image.resize(fixed_size)

        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        def rgb_to_string(rgb):
            return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
        string_array = np.apply_along_axis(rgb_to_string, 2, image_array)
        df = pd.DataFrame(string_array)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Uploaded Image")
            st.image(image, use_column_width=True)

        with col2:
            st.header(f"{image_type} - Image Matrix")
            st.dataframe(df,height=350,width=500)


        result_images = []

        with st.sidebar:
            with st.expander("Histogram Equalization"):
                if st.button("Apply Histogram Equalization"):
                    equalized_image = histogram_equalization(image)
                    result_images.append(equalized_image)

            with st.expander("Plot RGB Histogram"):
                if st.button("Plot RGB Histogram"):
                    hist_image = plot_histogram_as_image(image)
                    result_images.append(hist_image)

            with st.expander("Plot Grayscale Histogram"):
                if st.button("Plot Grayscale Histogram"):
                    gray_hist_image = plot_grayscale_histogram_as_image(image)
                    result_images.append(gray_hist_image)



            with st.expander("Canny Edge Detection"):
                low_threshold = st.slider("Low threshold:", 0, 255, 50)
                high_threshold = st.slider("High threshold:", 0, 255, 150)
                if st.button("Apply Canny"):
                    canny_image = canny_edge_detection(image, low_threshold, high_threshold)
                    result_images.append(canny_image)

            with st.expander("Sobel Edge Detection"):
                ksize = st.slider("Kernel size:", 1, 31, 3, step=2)
                if st.button("Apply Sobel"):
                    sobel_image = sobel_edge_detection(image, ksize)
                    result_images.append(sobel_image)

            with st.expander("Apply Gaussian Blur"):
                kernel_size = st.slider("Kernel size:", 1, 31, 5, step=2)
                sigma_x = st.slider("Sigma X:", 0.0, 10.0, 0.0)
                if st.button("Apply Gaussian Blur"):
                    blurred_image = apply_gaussian_blur(image, kernel_size=(kernel_size, kernel_size), sigma_x=sigma_x)
                    result_images.append(blurred_image)

            with st.expander("Extract HOG Features"):
                orientations = st.slider("Orientations:", 1, 16, 9)
                pixels_per_cell = st.slider("Pixels per Cell:", 4, 32, 8)
                cells_per_block = st.slider("Cells per Block:", 1, 5, 3)
                if st.button("Extract HOG Features"):
                    hog_features_image = extract_hog_features(image, orientations=orientations,
                                                              pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                                              cells_per_block=(cells_per_block, cells_per_block))
                    result_images.append(hog_features_image)


            with st.expander("Region Growing segmentation"):
                st.write("Click on the button to open the image in a separate window and select seed points.")
                if st.button("Select Seed Points"):
                    points = get_click_coordinates(image_array)
                    st.session_state['points'] = points
                    st.write(f"Selected points: {points}")

                threshold = st.slider("Threshold", min_value=1, max_value=255, value=30)

                if st.button("Apply Region Growing"):
                    if 'points' in st.session_state:
                        points = st.session_state['points']
                        grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                        for point in points:
                            region_growing_image = region_growing1(grayscale_image, [point], threshold)
                            result_images.append(region_growing_image)
                    else:
                        st.write("No points selected. Please select seed points first.")



            with st.expander("Apply Morphological Operation"):
                morph_operation = st.selectbox("Select Morphological Operation",
                                               ["Opening", "Closing", "Erosion", "Dilation"])
                morph_size = st.number_input("Enter size of structuring element:", min_value=1, value=3)
                threshold_morph = st.slider("Threshold for Morph Operation:", 0, 255, 128)
                if st.button("Apply Morphological Operation"):
                    if 'binary_image' not in locals():
                        binary_image = binary_conversion(image, threshold_morph)
                    morph_operation = morph_operation.lower()
                    morph_image = apply_morphological_operation(binary_image, operation=morph_operation,
                                                                size=morph_size)
                    result_images.append(morph_image)

            with st.expander("Convert to Grayscale"):
                if st.button("Convert to Grayscale"):
                    grayscale_image = grayscale_conversion(image)
                    result_images.append(grayscale_image)

            with st.expander("Convert to Binary"):
                threshold = st.slider("Threshold for binarization:", 0, 255, 128)
                if st.button("Convert to Binary"):
                    binary_image = binary_conversion(image, threshold)
                    result_images.append(binary_image)

            with st.expander("RGB Channel Separation"):
                if st.button("Show RGB Channels"):
                    red_image, green_image, blue_image = display_rgb_channels(image)
                    result_images.append(red_image)
                    result_images.append(green_image)
                    result_images.append(blue_image)

            with st.expander("Adjust Contrast"):
                contrast = st.slider("Contrast intensity:", 0.0, 2.0, 1.0)
                if st.button("Adjust Contrast"):
                    contrasted_image = adjust_contrast(image, contrast)
                    result_images.append(contrasted_image)

            with st.expander("Rotate Image"):
                angle = st.slider("Angle of rotation:", -180, 180, 0)
                if st.button("Rotate Image"):
                    rotated_image = rotate_image(image, angle)
                    result_images.append(rotated_image)

            with st.expander("Adjust Brightness"):
                brightness = st.slider("Brightness intensity:", 0.0, 2.0, 1.0)
                if st.button("Adjust Brightness"):
                    brightened_image = adjust_brightness(image, brightness)
                    result_images.append(brightened_image)

                # Canny edge detection expander

            with st.expander("Zoom Image"):
                zoom_factor = st.slider("Zoom factor:", 0.1, 3.0, 1.0)
                if st.button("Apply Zoom"):
                    zoomed_image = zoom_image(image, zoom_factor)
                    result_images.append(zoomed_image)

            with st.expander("Shift Image"):
                shift_x = st.slider("Shift along X-axis:", -100, 100, 0)
                shift_y = st.slider("Shift along Y-axis:", -100, 100, 0)
                if st.button("Apply Shift"):
                    shifted_image = shift_image(image, shift_x, shift_y)
                    result_images.append(shifted_image)

            with st.expander("Flip Image"):
                direction = st.selectbox("Flip direction:", ["Horizontal", "Vertical"])
                if st.button("Apply Flip"):
                    flipped_image = flip_image(image, direction)
                    result_images.append(flipped_image)

            with st.expander("Resize Image"):
                resize_width = st.number_input("Enter width for resizing:", min_value=1)
                resize_height = st.number_input("Enter height for resizing:", min_value=1)
                if st.button("Resize Image"):
                    resized_image = resize_image(image, resize_width, resize_height)
                    result_images.append((resized_image, "Resized Image", resize_width, resize_height))


        if result_images:
            if st.button("Clear Result Images"):
                st.session_state.result_images = []

            if len(result_images) ==3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(result_images[0], use_column_width=True)
                with col2:
                    st.image(result_images[1], use_column_width=True)
                with col3:
                    st.image(result_images[2], use_column_width=True)
                    
        for result_image in result_images:
            if isinstance(result_image, tuple):
                result_image, caption, width, height = result_image
                st.image(result_image, width=width)
            else:
                st.image(result_image, use_column_width=True)


if __name__ == "__main__":
    main()
