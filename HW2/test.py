import numpy as np

def label_components_4_connected(binary_image):
    labeled_image = np.zeros_like(binary_image)
    label = 1
    rows, cols = binary_image.shape

    print(binary_image)
    print(labeled_image)

    for i in range(rows):
        for j in range(cols):
            print(i, j)
            neighbors = []
            if binary_image[i, j] == 1:
                if i > 0:
                    neighbors.append(labeled_image[i - 1, j])
                if j > 0:
                    neighbors.append(labeled_image[i, j - 1])

                if len(neighbors) == 0 or all(neighbor == 0 for neighbor in neighbors):
                    labeled_image[i, j] = label
                    label += 1
                else:
                    labeled_image[i, j] = min(neighbor for neighbor in neighbors if neighbor != 0)
            print(neighbors)

    return labeled_image

def label_components_8_connected(binary_image):
    labeled_image = np.zeros_like(binary_image)
    label = 1
    rows, cols = binary_image.shape
    
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                neighbors = []
                if i > 0:
                    neighbors.append(labeled_image[i-1, j])
                    if j > 0:
                        neighbors.append(labeled_image[i-1, j-1])
                    if j < cols - 1:
                        neighbors.append(labeled_image[i-1, j+1])
                if j > 0:
                    neighbors.append(labeled_image[i, j-1])

                if len(neighbors) == 0 or all(neighbor == 0 for neighbor in neighbors):
                    labeled_image[i, j] = label
                    label += 1
                else:
                    labeled_image[i, j] = min(neighbor for neighbor in neighbors if neighbor != 0)

    return labeled_image

def label_components(binary_image, connectivity=4):
    # Create a copy of the binary image to store labels
    labeled_image = np.zeros_like(binary_image)
    
    # Initialize Union-Find data structure
    labels = {}
    next_label = 1
    
    # Define neighbors based on connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (0, -1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (-1, -1), (-1, 1), (0, -1)]
    else:
        raise ValueError("Connectivity should be either 4 or 8.")
    
    # First pass
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 1:
                neighbor_labels = []
                i = 0
                for dy, dx in neighbors:
                    nx, ny = x + dx, y + dy
                    print('y=', y, ', x=', x, ', ny=', ny, ', nx=', nx, ', i=', i)
                    i+=1
                    if 0 <= nx < binary_image.shape[1] and 0 <= ny < binary_image.shape[0]:
                        neighbor_label = labeled_image[ny, nx]
                        if neighbor_label != 0:
                            neighbor_labels.append(neighbor_label)
                
                if len(neighbor_labels) == 0:
                    labeled_image[y, x] = next_label
                    labels[next_label] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbor_labels)
                    labeled_image[y, x] = min_label
                    for label in neighbor_labels:
                        if label != min_label:
                            labels[label] = min_label
    
    # Second pass
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if labeled_image[y, x] != 0:
                root_label = labels[labeled_image[y, x]]
                while root_label != labels[root_label]:
                    root_label = labels[root_label]
                labeled_image[y, x] = root_label
    
    row, col = binary_image.shape
    print(row, col)
    print(binary_image.shape)

    return labeled_image

# Example usage:
binary_image = np.array([[0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 0],
                         [0, 0, 0, 1, 1],
                         [1, 1, 0, 0, 1],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 0]])

labeled_image_4_connected = label_components_4_connected(binary_image)
labeled_image_8_connected = label_components_8_connected(binary_image)

print("4-connected labeled image:")
print(labeled_image_4_connected)
print("\n8-connected labeled image:")
print(labeled_image_8_connected)

t4 = label_components(binary_image, 4)
t8 = label_components(binary_image, 8)
print("ttt4")
print(t4)
print("ttt8")
print(t8)
custom_colormap = np.array([
        [241, 199, 106],# Yellow
        [240, 230, 140],# Khaki
        [0, 0, 255],    # Blue
        [0, 255, 255],  # Cyan
        [222, 184, 135],# BurlyWood
        [255, 255, 224],# LightYellow
        [128, 0, 0],    # Maroon
        [0, 128, 0],    # Green
        [0, 0, 128],    # Navy
        [0, 128, 128],  # Teal
        [198, 110, 72], # Brown
        [128, 128, 0],  # Olive
        [0, 0, 0],      # Black
        [128, 128, 128],# Gray
        [192, 192, 192],# Silver
        [255, 255, 255] # White
    ])
print(custom_colormap[0])