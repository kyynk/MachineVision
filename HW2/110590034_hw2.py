'''
Modules import
'''
import numpy as np
import cv2

def rgb_to_gray(image):
    '''
    Convert color image to grayscale image
    '''
    gray_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])
    return gray_image.astype(np.uint8)

def gray_to_binary(gray_image, threshold):
    '''
    Convert grayscale image to binary image
    '''
    binary_image = np.where(gray_image > threshold, 0, 255)
    return binary_image.astype(np.uint8)

def label_components(binary_image, connectivity=4):
    '''
    Label connected components in a binary image using the classical algorithm.
    Connectivity type for labeling. Can be 4 or 8. Defaults to 4.
    Raises ValueError: If connectivity is not 4 or 8.
    '''
    labeled_image = np.zeros_like(binary_image, dtype=np.uint64)

    # Initialize Union-Find data structure
    labels = {}
    now_label = 1

    # Define neighbors based on connectivity
    # (y, x)
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (-1, -1), (-1, 1), (0, -1)]
    else:
        raise ValueError("Connectivity should be either 4 or 8.")

    print(connectivity)
    print(neighbors)
    print('bin=', np.unique(binary_image))
    rows, cols = binary_image.shape

    # First pass
    for now_y in range(rows):
        for now_x in range(cols):
            if binary_image[now_y, now_x] == 255:
                neighbor_labels = []

                for direction_y, direction_x in neighbors:
                    neighbor_y, neighbor_x = now_y + direction_y, now_x + direction_x

                    if 0 <= neighbor_y < rows and 0 <= neighbor_x < cols:
                        neighbor_label = labeled_image[neighbor_y, neighbor_x]

                        if neighbor_label != 0:
                            neighbor_labels.append(neighbor_label)

                if len(neighbor_labels) == 0:
                    labeled_image[now_y, now_x] = now_label
                    labels[now_label] = now_label
                    now_label += 1
                else:
                    min_label = min(neighbor_labels)
                    labeled_image[now_y, now_x] = min_label
                    for label in neighbor_labels:
                        if label != min_label:
                            root_label = labels[label]
                            while root_label != labels[root_label]:
                                root_label = labels[root_label]

                            if labels[min_label] == min_label:
                                labels[max(root_label, min_label)] = min(root_label, min_label)
                            else:
                                root_min_label = labels[min_label]
                                while root_min_label != labels[root_min_label]:
                                    root_min_label = labels[root_min_label]
                                labels[max(root_label, root_min_label)] = min(root_label, root_min_label)

    # Second pass
    for y in range(rows):
        for x in range(cols):
            if labeled_image[y, x] != 0:
                root_label = labels[labeled_image[y, x]]
                while root_label != labels[root_label]:
                    root_label = labels[root_label]
                labeled_image[y, x] = root_label

    # print('labels =', labels)
    return labeled_image

def label_to_color(labeled_image):
    '''
    Assign color to label
    '''
    unique_labels = np.unique(labeled_image)
    print(unique_labels)
    colors = np.random.randint(100, 256, (len(unique_labels), 3), dtype=np.uint8)
    colored_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3), dtype=np.uint8)
    for label, color in zip(unique_labels, colors):
        if label != 0:
            colored_image[labeled_image == label] = color
    return colored_image

def image(number, threshold):
    '''
    For img{number}.png
    '''
    image = cv2.imread(f'images/img{number}.png')
    gray_image = rgb_to_gray(image)
    binary_image = gray_to_binary(gray_image, threshold)

    labeled_image_4 = label_components(binary_image, connectivity=4)
    labeled_image_8 = label_components(binary_image, connectivity=8)

    colored_image_4 = label_to_color(labeled_image_4)
    colored_image_8 = label_to_color(labeled_image_8)
    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('4 Connected Components', colored_image_4)
    cv2.imshow('8 Connected Components', colored_image_8)
    cv2.imwrite(f'results/img{number}_4.png', colored_image_4)
    cv2.imwrite(f'results/img{number}_8.png', colored_image_8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, threshold=170)
    image(number=2, threshold=173)
    image(number=3, threshold=206)
    image(number=4, threshold=240)
