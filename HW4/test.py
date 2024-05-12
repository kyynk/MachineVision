import numpy as np
import cv2

class WatershedSegmentation:
    def __init__(self, image):
        self.image = image
        self.labels = np.zeros_like(image)
        self.current_label = 1
        self.marked_points = []
        self.queue = []
        self.neighborhood = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def mark_area(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.marked_points.append((x, y))
            cv2.circle(self.labels, (x, y), 5, (255, 255, 255), -1)

    def init_queue(self):
        for x, y in self.marked_points:
            self.queue.append((x, y))
            self.labels[x, y] = -2

    def grow_region(self):
        while self.queue:
            x, y = self.queue.pop(0)
            self.labels[x, y] = self.get_label(x, y)

            for dx, dy in self.neighborhood:
                nx, ny = x + dx, y + dy
                if self.is_valid_neighbor(nx, ny):
                    if self.labels[nx, ny] == 0:
                        self.labels[nx, ny] = -2
                        self.queue.append((nx, ny))
                    elif self.labels[nx, ny] != self.labels[x, y]:
                        self.labels[x, y] = -1

    def get_label(self, x, y):
        neighbor_labels = []
        for dx, dy in self.neighborhood:
            nx, ny = x + dx, y + dy
            if self.is_valid_neighbor(nx, ny) and self.labels[nx, ny] > 0:
                neighbor_labels.append(self.labels[nx, ny])
        if len(neighbor_labels) == 0:
            return self.labels[x, y]
        elif len(set(neighbor_labels)) == 1:
            return neighbor_labels[0]
        else:
            return -1

    def is_valid_neighbor(self, x, y):
        return 0 <= x < self.image.shape[0] and 0 <= y < self.image.shape[1]

    def segment(self):
        self.init_queue()
        self.grow_region()

    def display_segmented_image(self):
        cv2.imshow('Segmented Image', self.labels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load your image here
    image = cv2.imread('images/img1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    segmentation = WatershedSegmentation(gray)
    cv2.namedWindow('Original Image')
    cv2.setMouseCallback('Original Image', segmentation.mark_area)
    cv2.imshow('Original Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    segmentation.segment()
    segmentation.display_segmented_image()
