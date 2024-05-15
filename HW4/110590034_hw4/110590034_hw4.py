'''
Modules import
'''
import numpy as np
import cv2

from heapq import heappush, heappop

def mark_on_image(image_path, number):
    '''
    Mark on image by mouse event
    '''
    def draw_circle(event, x, y, flags, param):
        '''
        Draw circle
        '''
        nonlocal img, drawing, radius, color_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(img, (x, y), radius, colors[number-1][color_idx], -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(img, (x, y), radius, colors[number-1][color_idx], -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    img = cv2.imread(image_path)

    # Create a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    drawing = False  # True if mouse is pressed
    radius = 3  # Initial radius
    colors = [
                [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0)],
                [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 255), (0, 255, 255), (255, 140, 0), (128, 128, 128),
                 (128, 0, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192),
                 (255, 165, 0), (255, 192, 203)],
                [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
             ]
    color_idx = 0  # Initial color index

    while True:
        img_with_text = img.copy()
        # Display current color
        cv2.putText(img_with_text, f'Color: {colors[number-1][color_idx]}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('image', img_with_text)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Press 'q' to quit
            break
        elif k == ord('c'):  # Press 'c' to clear canvas
            img = cv2.imread(image_path)
        elif k == ord('+'):  # Increase radius
            radius += 1
        elif k == ord('-'):  # Decrease radius (minimum is 1)
            radius = max(1, radius - 1)
        elif k == ord('n'):  # Change color (next)
            color_idx = (color_idx + 1) % len(colors[number-1])
        elif k == ord('s'):  # Save image
            cv2.imwrite(f'results\img{number}_q1-1.png', img)
            print("Image saved.")

    cv2.destroyAllWindows()

def calculate_priority(y, x, origin_image, seeds, method=1):
    '''
    Calculate priority
    '''
    height = origin_image.shape[0]
    width = origin_image.shape[1]
    priority = 0
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= y+i < height and 0 <= x+j < width\
               and (i != 0 or j != 0):
                neighbors.append((y+i, x+j))
    neighbors_rgb = [list(origin_image[i][j]) for i, j in neighbors]
    neighbors_rgb = np.array(neighbors_rgb)
    mean_rgb = np.mean(neighbors_rgb, axis=0)
    variance_rgb = np.var(neighbors_rgb, axis=0)

    # first priority method
    if method == 1:
        priority = 0.5 * mean_rgb.sum() + 0.5 * variance_rgb.sum()
    # second priority method
    elif method == 2:
        neighbors_distance = np.linalg.norm(neighbors_rgb - origin_image[y][x], axis=1)
        seeds_rgb = [list(origin_image[i][j]) for i, j in seeds]
        seeds_rgb = np.array(seeds_rgb)
        seeds_distance = np.linalg.norm(seeds_rgb - origin_image[y][x], axis=1)
        priority = 0.5 * mean_rgb.sum() + 0.5 * variance_rgb.sum() + 0.5 * np.min(neighbors_distance) + 0.75 * np.min(seeds_distance)
    # third priority method
    else:
        seeds_rgb = [list(origin_image[i][j]) for i, j in seeds]
        seeds_rgb = np.array(seeds_rgb)
        seeds_distance = np.linalg.norm(seeds_rgb - origin_image[y][x], axis=1)
        priority = 0.5 * variance_rgb.sum() + 0.5 * np.min(seeds_distance)
    # print(priority)
    return priority

def colorize_watershed(origin_image, label_map, colors, number):
    '''
    colorize watershed
    '''
    height = origin_image.shape[0]
    width = origin_image.shape[1]
    o_img = np.array(origin_image).astype(np.float64)
    colors = np.array(colors[number-1]).astype(np.float64)
    region_image = np.zeros((height, width, 3))
    for row in range(height):
        for col in range(width):
            if label_map[row][col] > 0:
                region_image[row][col] = colors[label_map[row][col]-1]
            else:
                region_image[row][col] = [0, 0, 0]
    watershed_image = np.zeros((height, width, 3))
    region_image = region_image.astype(np.float64)
    print(colors)
    print(np.unique(label_map))
    for row in range(height):
        for col in range(width):
            if label_map[row][col] > 0:
                watershed_image[row][col] = o_img[row][col] * 0.5 + region_image[row][col] * 0.5
    cv2.imshow('Region Image', region_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return watershed_image

def watershed(origin_image, marked_image, number, priority_method=1):
    '''
    watershed
    '''
    colors = [
        [[255, 0, 0], [0, 128, 0], [0, 0, 255], [255, 255, 0]],
        [[255, 0, 0], [0, 128, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 140, 0],
         [128, 128, 128], [128, 0, 0], [128, 0, 128], [0, 128, 128], [192, 192, 192], [255, 165, 0], [255, 192, 203]],
        [[255, 0, 0], [0, 128, 0], [0, 0, 255]]
    ]
    height = origin_image.shape[0]
    width = origin_image.shape[1]
    label_map = np.zeros((height, width), dtype=np.int64)
    print(label_map.shape[0])
    print(label_map.shape[1])
    seeds = []
    # 1-1 mark area
    for row in range(height):
        for col in range(width):
            if list(marked_image[row][col]) in colors[number-1]:
                label_map[row][col] = colors[number-1].index(list(marked_image[row][col])) + 1
                seeds.append((row, col))
    print('mark area done')
    # print(seeds)
    # 1-2 region growing
    # create priority queue needed for region growing
    priority_queue = []
    for seed in seeds:
        heappush(priority_queue, (calculate_priority(seed[0], seed[1], origin_image, seeds, priority_method), seed))
    print('add seeds in priority queue done')
    neighbors_direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    count  = 0
    print('start region growing:')
    while len(priority_queue) != 0:
        priority, (y, x) = heappop(priority_queue)
        count += 1
        print(count)
        neighbors = []
        for direction in neighbors_direction:
            n_y = y + direction[0]
            n_x = x + direction[1]
            if 0 <= n_y < height and 0 <= n_x < width:
                neighbors.append((n_y, n_x))
        neighbors_label = [label_map[i][j] for i, j in neighbors]
        unique_mark_label = [label for label in np.unique(neighbors_label) if label > 0]
        if label_map[y][x] == -2:
            # mark label count > 2 or = 0
            if len(unique_mark_label) > 1 or len(unique_mark_label) == 0:
                label_map[y][x] = -1
                continue
            if len(unique_mark_label) == 1:
                label_map[y][x] = unique_mark_label[0]
        for i, j in neighbors:
            if label_map[i][j] == 0:
                label_map[i][j] = -2
                heappush(priority_queue, (calculate_priority(i, j, origin_image, seeds, priority_method), (i, j)))

    watershed_image = colorize_watershed(origin_image, label_map, colors, number)
    return watershed_image

def image(number, priority_method=1):
    '''
    For img{number}.png
    '''
    origin_image = cv2.imread(f'images/img{number}.jpg')
    mark_on_image(f'images/img{number}.jpg', number)
    marked_image = cv2.imread(f'results/img{number}_q1-1.png')
    watershed_image = watershed(origin_image, marked_image, number, priority_method)
    cv2.imwrite(f'results/img{number}_q1.jpg', watershed_image)
    segmented_image = cv2.imread(f'results/img{number}_q1.jpg')

    cv2.imshow('Origin Image', origin_image)
    cv2.imshow('Marked Image', marked_image)
    cv2.imshow('Watershed Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1, priority_method=3)
    image(number=2, priority_method=3)
    image(number=3, priority_method=2)
