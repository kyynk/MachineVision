import numpy as np

def distance_transform_8(image):
	height, width = image.shape
	distance_map = np.zeros_like(image, dtype=np.uint16)
	distance_map = np.where(image == 255, 1, 0)

	print(distance_map)
	top_left_neighbors = [(-1, 0), (-1, -1), (-1, 1), (0, -1)]


	for i in range(height):
		for j in range(width):
			if image[i, j] == 0:
				continue
			neighbors = []
			for n_y, n_x in top_left_neighbors:
				if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
					neighbors.append(distance_map[i + n_y, j + n_x])
			print(len(neighbors))
			if neighbors:
				distance_map[i, j] = min(neighbors) + 1
	
	print(distance_map)
	bottom_right_neighbors = [(1, 0), (1, 1), (1, -1), (0, 1)]

	for i in range(height - 1, -1, -1):
		for j in range(width - 1, -1, -1):
			if image[i, j] == 0:
				continue
			neighbors = []
			neighbors.append(distance_map[i, j])
			for n_y, n_x in bottom_right_neighbors:
				if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
					neighbors.append(distance_map[i + n_y, j + n_x])
			print(len(neighbors), neighbors)
			if neighbors:
				distance_map[i, j] = min(neighbors) + 1

	return distance_map

image = np.array([[0, 0, 0, 0, 0],
				  [0, 255, 255, 0, 0],
				  [0, 255, 255, 0, 0],
				  [0, 0, 0, 255, 0],
				  [0, 0, 0, 0, 0]], dtype=np.uint8)

# print(distance_transform_8(image))
# image_square = np.array([[1, 1, 1, 1, 1],
#                          [1, 0, 1, 1, 1],
#                          [1, 0, 1, 1, 1],
#                          [1, 0, 0, 0, 1],
#                          [1, 1, 1, 1, 1]], dtype=np.uint8)


image_square = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 255, 255, 0, 0, 255, 255, 0, 0, 0],
                         [0, 255, 0, 0, 0, 0, 0, 0, 255, 0],
                         [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                         [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                         [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
						 [0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                         [0, 255, 0, 0, 0, 0, 0, 0, 255, 0],
                         [0, 255, 255, 0, 0, 255, 255, 255, 0, 0],
						 [0, 255, 255, 0, 0, 255, 255, 255, 0, 0],
						 [0, 255, 255, 0, 0, 255, 255, 255, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)


print(distance_transform_8(image_square))