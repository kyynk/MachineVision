import numpy as np

def distance_transform(image, connectivity=4):
    '''
    Distance transform with 8 neighbors
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
            neighbors.append(distance_map[i, j])
            for n_y, n_x in bottom_right_neighbors:
                if 0 <= (i + n_y) < height and 0 <= (j + n_x) < width:
                    neighbors.append(distance_map[i + n_y, j + n_x])
            # print(len(neighbors), neighbors)
            if neighbors:
                distance_map[i, j] = min(min(neighbors) + 1, distance_map[i, j])

    return distance_map

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

def find_local_maximum_connectivity(distance_image):
    '''
    Find local maximum and connectivity image
    '''
    local_maximum_connectivity_image = np.copy(distance_image)
    rows, cols = distance_image.shape
    
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
    
    # local_maximum_connectivity_image = np.where(local_maximum_connectivity_image > 0, 255, 0).astype(np.uint8)
    return local_maximum_connectivity_image

def medial_axis(distance_image):
    '''
    Medial axis
    '''
    local_maximum = find_local_maximum_connectivity(distance_image)
    return local_maximum

def zhang_suen_thinning(image):
    
    def get_neighbors(y, x, image):
        '''
        [p2, p3, p4, p5,
        p6, p7, p8, p9]
        '''
        return [image[y-1, x], image[y-1, x+1], image[y, x+1], image[y+1, x+1],
                image[y+1, x], image[y+1, x-1], image[y, x-1], image[y-1, x-1]]

    def get_not_zero_neighbors_count(neighbors):
        not_zero_count = 0
        for distance in neighbors:
            if distance != 0:
                not_zero_count += 1
        return not_zero_count

    def get_transition(neighbors):
        if neighbors[0]:
            is_zero = False
        else:
            is_zero = True
        transition_count = 0
        for i in range(1, len(neighbors)+1):
            distance = neighbors[i%len(neighbors)]
            if is_zero and distance != 0:
                transition_count += 1
                is_zero = False
            elif distance:
                is_zero = False
            else:
                is_zero = True
        return transition_count

    def thinning(image):
        '''
        p9 p2 p3
        p8 p1 p4
        p7 p6 p5
        '''
        skeleton = np.copy(image)
        rows, cols = image.shape
        change1 = [(-1, -1)]
        change2 = [(-1, -1)]
        while change1 or change2:
            change1 = []
            for y in range(1, rows-1):
                for x in range(1, cols-1):
                    if skeleton[y, x] != 0:
                        neighbors = get_neighbors(y, x, skeleton)
                        not_zero_neighbors_count = get_not_zero_neighbors_count(neighbors)
                        transition = get_transition(neighbors)
                        if (2 <= not_zero_neighbors_count <= 6 and\
                            transition == 1 and\
                            neighbors[2-2] * neighbors[4-2] * neighbors[6-2] == 0 and\
                            neighbors[4-2] * neighbors[6-2] * neighbors[8-2] == 0):
                            change1.append((y, x))
            for y, x in change1:
                skeleton[y, x] = 0
            change2 = []
            for y in range(1, rows-1):
                for x in range(1, cols-1):
                    if skeleton[y, x] != 0:
                        neighbors = get_neighbors(y, x, skeleton)
                        not_zero_neighbors_count = get_not_zero_neighbors_count(neighbors)
                        transition = get_transition(neighbors)
                        if (2 <= not_zero_neighbors_count <= 6 and\
                            transition == 1 and\
                            neighbors[2-2] * neighbors[4-2] * neighbors[8-2] == 0 and\
                            neighbors[2-2] * neighbors[6-2] * neighbors[8-2] == 0):
                            change2.append((y, x))
            for y, x in change2:
                skeleton[y, x] = 0
        skeleton = np.where(skeleton > 0, 255, 0).astype(np.uint8)
        return skeleton

    return thinning(image)

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

a = np.array([[1, 1, 1],
          [1, 0, 1],
          [0, 0, 1]])
print(sum(a[0]))
print(sum(a[:, 0]))
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
                         [0, 255, 255, 0, 0, 255, 255, 0, 0, 0],
						 [0, 255, 255, 0, 0, 255, 255, 255, 0, 0],
						 [0, 255, 255, 0, 0, 255, 255, 255, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# test = np.copy(image_square)
# test[1, 1] = 1
# print(image_square)
# print(test)

distance = distance_transform(image_square, 4)
print(distance)
print('-' * 100)
medial_axisa = medial_axis(distance)
print(medial_axisa)
# print('-' * 100)
# skeleton = zhang_suen_thinning(find_local_maximum_connectivity(distance))
# print(skeleton)