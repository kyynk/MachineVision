import numpy as np

def distance_transform_8(image):
	height, width = image.shape
	distance_map = np.zeros_like(image, dtype=np.uint16)

	for i in range(height):
		for j in range(width):
			if image[i, j] != 0:
				distance_map[i, j] = 1
			else:
				distance_map[i, j] = 0

	print(distance_map)

	for m in range(1, max(height, width)):
		print(m)
		for i in range(height):
			for j in range(width):
				if image[i, j] == 0:
					continue
				neighbors = []
				for u in range(max(0, i-1), min(height, i+2)):
					for v in range(max(0, j-1), min(width, j+2)):
						neighbors.append(distance_map[u, v])
				print(neighbors)
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


image_square = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]], dtype=np.uint8)


print(distance_transform_8(image_square))