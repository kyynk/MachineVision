'''
Modules import
'''
import numpy as np
import cv2

from heapq import heappush, heappop

def mark_on_image(image_path, number):
    '''
    mark on image
    '''
    def draw_circle(event, x, y, flags, param):
        '''
        draw circle
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
                 (255, 0, 255), (0, 255, 255), (0, 0, 0), (128, 128, 128),
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
            cv2.imwrite(f'results\img{number}_q1-1.jpg', img)
            print("Image saved.")

    cv2.destroyAllWindows()

def watershed(origin_image, marked_image, number):
    '''
    watershed
    '''
    colors = [
        [[255, 0, 0], [0, 128, 0], [0, 0, 255], [255, 255, 0]],
        [[255, 0, 0], [0, 128, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0],
         [128, 128, 128], [128, 0, 0], [128, 0, 128], [0, 128, 128], [192, 192, 192], [255, 165, 0], [255, 192, 203]],
        [[255, 0, 0], [0, 128, 0], [0, 0, 255]]
    ]
    height = origin_image.shape[0]
    width = origin_image.shape[1]
    label_map = np.zeros((height, width, 3), dtype=np.float64)
    seeds = []
    # 1-1 mark area
    for row in range(height):
        for col in range(width):
            if list(marked_image[row][col]) in colors[number-1]:
                label_map[row][col] = colors[number-1].index(list(marked_image[row][col])) + 1
                seeds.append((row, col))

    # 1-2 region growing
    # create priority queue needed for region growing
    priority_queue = []


def image(number):
    '''
    For img{number}.png
    '''
    origin_image = cv2.imread(f'images/img{number}.jpg')
    # mark_on_image(f'images/img{number}.jpg', number)
    marked_image = cv2.imread(f'results/img{number}_q1-1.jpg')
    watershed(origin_image, marked_image, number)

    # cv2.imshow('Marked Image', marked_image)
    # cv2.imwrite(f'results/img{number}_q1.jpg', colors_distance_image_8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image(number=1)
    # image(number=2)
    # image(number=3)
