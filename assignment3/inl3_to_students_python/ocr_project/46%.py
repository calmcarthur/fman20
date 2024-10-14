def segment2feature(segment):
    # Translate segment to center based on center of mass (C.O.M)
    rows, columns = np.nonzero(segment)
    center_of_mass_x = np.mean(columns)
    center_of_mass_y = np.mean(rows)

    # Calculate the shift in x and y directions to align with the center
    shift_x = (segment.shape[1] // 2) - center_of_mass_x
    shift_y = (segment.shape[0] // 2) - center_of_mass_y

    # Perform the translation using a transformation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(segment, M, (segment.shape[1], segment.shape[0]))

    features = []

    # # Feature 1: Image width (number of columns with non-zero pixels)
    width = np.sum(np.any(translated, axis=0))
    features.append(width)

    # # Feature 2: Top-heaviness (upper half pixel sum / total pixel sum)
    height = translated.shape[0]
    # top_heaviness = np.sum(translated[:height//2]) / np.sum(translated)
    # features.append(top_heaviness)

    # Feature 3: Right-heaviness (right half pixel sum / total pixel sum)
    width = translated.shape[1]
    right_heaviness = np.sum(translated[:, width//2:]) / np.sum(translated)
    features.append(right_heaviness)

    # Feature 4: Number of holes (connected components in inverted image)
    inverted_segment = cv2.bitwise_not(translated)
    num_holes, _ = cv2.connectedComponents(inverted_segment)
    # features.append(num_holes)

    # Feature 5: Holes inverse (1 / number of holes)
    # holes_inverse = 1 / num_holes
    # features.append(holes_inverse)

    # Feature 6: Sum of pixel values in four quadrants
    half_h, half_w = height // 2, width // 2
    top_left = np.sum(translated[:half_h, :half_w])
    top_right = np.sum(translated[:half_h, half_w:])
    bottom_left = np.sum(translated[half_h:, :half_w])
    bottom_right = np.sum(translated[half_h:, half_w:])
    features.extend([top_left, top_right, bottom_left, bottom_right])

    # Feature 7: Vertical symmetry (symmetric pixels along the y-axis)
    # vertical_symmetry = np.sum(translated == np.flip(translated, axis=0)) / translated.size
    # features.append(vertical_symmetry)

    # Feature 8: Horizontal symmetry (symmetric pixels along the x-axis)
    horizontal_symmetry = np.sum(translated == np.flip(translated, axis=1)) / translated.size
    features.append(horizontal_symmetry)

    # Feature 9: Mean of pixel values in the middle row
    # middle_row_mean = np.mean(translated[translated.shape[0] // 2, :])
    # features.append(middle_row_mean)

    # Feature 10: Mean of pixel values in the middle column
    # middle_col_mean = np.mean(translated[:, translated.shape[1] // 2])
    # features.append(middle_col_mean)

    # Feature 11: Division after erosion (1 / number of connected components after erosion)
    # eroded = cv2.erode(translated.astype(np.uint8), None, iterations=1)
    # num_components, _ = cv2.connectedComponents(eroded)
    # division_after_erosion = 1 / num_components
    # features.append(division_after_erosion)

    # Feature 12: Aspect ratio of the bounding box (width / height)
    _, _, w, h = cv2.boundingRect(np.column_stack((columns, rows)))
    # box_aspect_ratio = w / h
    # features.append(box_aspect_ratio)

    # Feature 13: Area cover (pixel sum / bounding box area)
    # area_cover_total = np.sum(segment) / (w * h)
    # features.append(area_cover_total)

    # Feature 14: Area to convex hull ratio (pixel sum / convex hull area)
    convex_hull = cv2.convexHull(np.column_stack((columns, rows)))
    convex_area = cv2.contourArea(convex_hull)
    area_convex_ratio = np.sum(segment) / convex_area
    features.append(area_convex_ratio)

    # Normalize all features together
    features = np.array(features)
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

    # Return final normalized feature array
    return normalized_features.reshape(-1, 1)