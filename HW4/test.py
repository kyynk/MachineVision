# import numpy as np
# import cv2

# class WatershedSegmentation:
#     def __init__(self, image):
#         self.image = image
#         self.labels = np.zeros_like(image, dtype=np.int8)
#         self.current_label = 1
#         self.marked_points = []
#         self.queue = []
#         self.neighborhood = [(1, 0), (-1, 0), (0, 1), (0, -1)]

#     def mark_area(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:  # Check bounds
#                 self.marked_points.append((x, y))
#                 cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)  # Draw on the original image

#     def init_queue(self):
#         for x, y in self.marked_points:
#             self.queue.append((x, y))
#             self.labels[x, y] = -2

#     def grow_region(self):
#         while self.queue:
#             x, y = self.queue.pop(0)
#             self.labels[x, y] = self.get_label(x, y)

#             for dx, dy in self.neighborhood:
#                 nx, ny = x + dx, y + dy
#                 if self.is_valid_neighbor(nx, ny) and (self.labels[nx, ny] == 0).any():
#                     self.labels[nx, ny] = -2
#                     self.queue.append((nx, ny))
#                 elif self.is_valid_neighbor(nx, ny) and (self.labels[nx, ny] != self.labels[x, y]).any():
#                     self.labels[x, y] = -1

#     def get_label(self, x, y):
#         neighbor_labels = []
#         for dx, dy in self.neighborhood:
#             nx, ny = x + dx, y + dy
#             if self.is_valid_neighbor(nx, ny) and (self.labels[nx, ny] > 0).any():
#                 neighbor_labels.append(self.labels[nx, ny])
#         if len(neighbor_labels) == 0:
#             return self.labels[x, y]
#         elif len(set(neighbor_labels)) == 1:
#             return neighbor_labels[0]
#         else:
#             return -1

#     def is_valid_neighbor(self, x, y):
#         return 0 <= x < self.image.shape[0] and 0 <= y < self.image.shape[1]

#     def segment(self):
#         self.init_queue()
#         self.grow_region()

#     def display_segmented_image(self):
#         cv2.imshow('Segmented Image', self.labels)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Load your image here
#     image = cv2.imread('images/img1.jpg')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     segmentation = WatershedSegmentation(image)  # Pass original image
#     cv2.namedWindow('Original Image')
#     cv2.setMouseCallback('Original Image', segmentation.mark_area)
#     cv2.imshow('Original Image', image)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     segmentation.segment()
#     segmentation.display_segmented_image()

import cv2
import numpy as np

# # Function to draw circle
# def draw_circle(event, x, y, flags, param):
#     global drawing, color_idx
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         cv2.circle(img, (x, y), radius, colors[color_idx], -1)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             cv2.circle(img, (x, y), radius, colors[color_idx], -1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False

# # Load an image
# img = cv2.imread('images/img1.jpg')

# # Create a window and bind the function to window
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_circle)

# drawing = False  # True if mouse is pressed
# radius = 10  # Initial radius
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # List of colors (blue, green, red)
# color_names = ['Blue', 'Green', 'Red']  # List of color names
# color_idx = 0  # Initial color index

# while True:
#     img_with_text = img.copy()
#     # Display current color
#     cv2.putText(img_with_text, f'Color: {color_names[color_idx]} ({colors[color_idx]})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.imshow('image', img_with_text)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('q'):  # Press 'q' to quit
#         break
#     elif k == ord('c'):  # Press 'c' to clear canvas
#         img = cv2.imread('images/img1.jpg')
#     elif k == ord('z'):  # Increase radius
#         radius += 1
#     elif k == ord('x'):  # Decrease radius (minimum is 1)
#         radius = max(1, radius - 1)
#     elif k == ord('n'):  # Change color (next)
#         color_idx = (color_idx + 1) % len(colors)
#     elif k == ord('s'):  # Save image
#         cv2.imwrite('results/img1_q1-1.jpg', img)
#         print("Image saved as 'drawn_image.jpg'.")

# cv2.destroyAllWindows()

import cv2
import numpy as np

def draw_on_image(image_path, number):
    # Function to draw circle
    def draw_circle(event, x, y, flags, param):
        nonlocal img, drawing, radius, color_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(img, (x, y), radius, colors[color_idx], -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(img, (x, y), radius, colors[color_idx], -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # Load an image
    img = cv2.imread(image_path)

    # Create a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    drawing = False  # True if mouse is pressed
    radius = 5  # Initial radius
    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (0, 0, 0), (128, 128, 128), (128, 0, 0), (128, 0, 128), (0, 128, 128), (192, 192, 192),
              (255, 165, 0), (255, 192, 203)]
    color_idx = 0  # Initial color index

    while True:
        img_with_text = img.copy()
        # Display current color
        cv2.putText(img_with_text, f'Color: {colors[color_idx]}', 
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
            color_idx = (color_idx + 1) % len(colors)
        elif k == ord('s'):  # Save image
            cv2.imwrite(f'results\img{number}_q1-1.jpg', img)
            print("Image saved.")

    cv2.destroyAllWindows()

# number = 1
# # Example usage:
# draw_on_image(f'images/img{number}.jpg', number)
# number = 2
# draw_on_image(f'images/img{number}.jpg', number)
lll = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [12, 10, 11]
]
print(np.array(lll).shape)
a = np.array(lll)
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))