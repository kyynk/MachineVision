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

def gray_to_binary(image, threshold):
    '''
    Convert image to binary image
    '''
    binary_image = np.where(image > threshold, 0, 255)
    return binary_image.astype(np.uint8)

def distance_transform(image, connectivity=4):
    '''
    Distance transform with 4, 8 neighbors
    '''
    height, width = image.shape
    distance_map = np.zeros_like(image, dtype=np.uint16)
    distance_map = np.where(image == 255, 1, 0)

    # print(distance_map)
    if connectivity == 4:
        top_left_neighbors = [(-1, 0), (0, -1)]
    else:
        top_left_neighbors = [(-1, 0), (-1, -1), (-1, 1), (0, -1)]

    for i in range(height):
        for j in range(width):
            if image[i, j] == 0:
                continue
            neighbors = []
            for n_y, n_x in top_left_neighbors:
                if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
                    neighbors.append(distance_map[i + n_y, j + n_x])
            # print(len(neighbors))
            if neighbors:
                distance_map[i, j] = min(neighbors) + 1
    
    # print(distance_map)
    if connectivity == 4:
        bottom_right_neighbors = [(1, 0), (0, 1)]
    else:
        bottom_right_neighbors = [(1, 0), (1, 1), (1, -1), (0, 1)]

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if image[i, j] == 0:
                continue
            neighbors = []
            for n_y, n_x in bottom_right_neighbors:
                if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
                    neighbors.append(distance_map[i + n_y, j + n_x])
            # print(len(neighbors), neighbors)
            if neighbors:
                distance_map[i, j] = min(min(neighbors) + 1, distance_map[i, j])

    return distance_map

def colorize_distance_transform(distance_image):
    '''
    Colorize distance transform image
    '''
    max_value = np.max(distance_image)
    colors_distance_image = (distance_image / max_value * 255).astype(np.uint8)
    return colors_distance_image

def is_connect_point(neighbors):
    '''
    Check connectivity
    '''
    # row 1, row 3
    if sum(neighbors[0]) > 0 and sum(neighbors[2]) > 0 and sum(neighbors[1]) == 0:
        return True
    # col 1, col 3
    if sum(neighbors[:, 0]) > 0 and sum(neighbors[:, 2]) > 0 and sum(neighbors[:, 1]) == 0:
        return True
    # neighbors which is lonely
    neighbor_direction = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                          (1, 0), (1, -1), (0, -1), (-1, -1)]
    for y in range(3):
        for x in range(3):
            if neighbors[y, x] != 0:
                n_neighbors = []
                for n_y, n_x in neighbor_direction:
                    if 0 <= y+n_y < 3 and 0 <= x+n_x < 3:
                        n_neighbors.append(neighbors[y+n_y, x+n_x])
                if sum(n_neighbors) == 0:
                    return True
    # hw example
    if neighbors[0, 0] != 0 and neighbors[1, 2] != 0 and neighbors[2, 1] != 0 and\
       neighbors[0, 1] == 0 and neighbors[1, 0] == 0:
        return True
    if neighbors[0, 2] != 0 and neighbors[1, 0] != 0 and neighbors[2, 1] != 0 and\
       neighbors[0, 1] == 0 and neighbors[1, 2] == 0:
        return True
    if neighbors[0, 1] != 0 and neighbors[1, 2] != 0 and neighbors[2, 0] != 0 and\
       neighbors[1, 0] == 0 and neighbors[2, 1] == 0:
        return True
    if neighbors[0, 1] != 0 and neighbors[1, 0] != 0 and neighbors[2, 2] != 0 and\
       neighbors[1, 2] == 0 and neighbors[2, 1] == 0:
        return True

    return False

def medial_axis(distance_image):
    '''
    Medial axis
    '''
    local_maximum_connectivity_image = np.copy(distance_image)
    rows, cols = distance_image.shape
    
    # find local maximum and keep connectivity
    for label in range(1, np.max(distance_image)+1):
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if distance_image[i, j] == label:
                    neighborhood = []
                    kernel = np.zeros((3, 3), dtype=np.uint16)
                    for u in range(i-1, i+2):
                        for v in range(j-1, j+2):
                            if u != i or v != j:
                                neighborhood.append((u, v))
                                kernel[u-i+1, v-j+1] = local_maximum_connectivity_image[u, v]
                            else:
                                kernel[u-i+1, v-j+1] = 0
                    max_local_distance = max([distance_image[u, v] for u, v in neighborhood])

                    if distance_image[i, j] < max_local_distance and not is_connect_point(kernel):
                        local_maximum_connectivity_image[i, j] = 0

    # remove extra redundancy for:
    # 1 1 0
    # 0 1 1
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if (distance_image[i, j-1] != 0 and distance_image[i, j] != 0) or\
               (distance_image[i, j] != 0 and distance_image[i, j+1] != 0) or\
               (distance_image[i-1, j] != 0 and distance_image[i, j] != 0) or\
               (distance_image[i, j] != 0 and distance_image[i+1, j] != 0) :
                kernel = np.zeros((3, 3), dtype=np.uint16)
                for u in range(i-1, i+2):
                    for v in range(j-1, j+2):
                        if u != i or v != j:
                            kernel[u-i+1, v-j+1] = local_maximum_connectivity_image[u, v]
                        else:
                            kernel[u-i+1, v-j+1] = 0
                if not is_connect_point(kernel):
                    local_maximum_connectivity_image[i, j] = 0

    skeleton_image = np.where(local_maximum_connectivity_image > 0, 255, 0).astype(np.uint8)
    return skeleton_image

def image(number, threshold):
    '''
    For img{number}.png
    '''
    image = cv2.imread(f'images/img{number}.jpg')
    gray_image = rgb_to_gray(image)
    binary_image = gray_to_binary(gray_image, threshold)
    distance_image_4 = distance_transform(binary_image, connectivity=4)
    distance_image_8 = distance_transform(binary_image, connectivity=8)

    colors_distance_image_4 = colorize_distance_transform(distance_image_4)
    colors_distance_image_8 = colorize_distance_transform(distance_image_8)
    skeleton_image = medial_axis(distance_image_4)

    cv2.imshow('Original Image', image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Binary Image', binary_image)
    cv2.imshow('Distance Transform Image With 4 Connectivity', colors_distance_image_4)
    cv2.imshow('Distance Transform Image With 8 Connectivity', colors_distance_image_8)
    cv2.imshow('Skeleton Image', skeleton_image)
    cv2.imwrite(f'results/img{number}_q1-1_4.jpg', colors_distance_image_4)
    cv2.imwrite(f'results/img{number}_q1-1_8.jpg', colors_distance_image_8)
    cv2.imwrite(f'results/img{number}_q1-2.jpg', skeleton_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, threshold=200)
    image(number=2, threshold=200)
    image(number=3, threshold=200)
    image(number=4, threshold=200)
