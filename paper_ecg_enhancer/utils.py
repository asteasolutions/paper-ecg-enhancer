from math import tan
import numpy as np
import cv2

def normalize_channel(image, ksize):
  """Normalizes the given single-channel image based on local color brightness."""
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
  close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  div = np.float32(image) / (close)
  result = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
  return result

def normalize_multi_channel(image, ksize):
  """Normalizes the given multi-channel image based on local color brightness.

  This method basically splits the image into its BGR channels, normalizes each channel separately
  using :py:func:`normalize_channel`, and merges back the normalized channels."""
  img = cv2.GaussianBlur(image, (5, 5), 0)
  (blue, green, red) = cv2.split(img)
  blue = normalize_channel(blue, ksize)
  green = normalize_channel(green, ksize)
  red = normalize_channel(red, ksize)
  return cv2.merge([blue, green, red])

def adjust_levels(image, sampling_region = None, bin_size = 4, low_percentile = 0.01, high_percentile = 0.75):
  """Adjusts the brightness levels in the given single-channel image based on the color distribution
  inside the specified sampling region.

  The color corresponding to low_percentile in the histogram of the sampling region is set to black,
  and the one corresponding to high_percentile is set to white. All other colors are rescaled
  uniformly in between."""
  region = image[sampling_region] if sampling_region is not None else image

  hist = cv2.calcHist([region], [0], None, [256 // bin_size], [0, 256]).ravel() / (region.shape[0] * region.shape[1])
  low = np.argmax(hist.cumsum() > low_percentile) * bin_size + (bin_size // 2)
  high = np.argmax(hist.cumsum() > high_percentile) * bin_size + (bin_size // 2)
  return ((image.clip(low, high) - low) * (255 / (high - low))).astype(np.uint8)

def order_points(pts):
  """Orders the given four-point array in clockwise direction starting from top-left, i.e.:
  [top-left, top-right, bottom-right, bottom-left]."""
  rect = np.zeros((4, 2), dtype='float32')

  sum = np.sum(pts, axis = 1)
  rect[0] = pts[np.argmin(sum)]
  rect[2] = pts[np.argmax(sum)]

  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]

  return rect

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
  """Computes a perspective transform, which converts the given quadrilateral to a rectangle
  with sides parralel to the coordinate axes. The sizes of the larger horizontal and vertical
  sides of the quadrilateral are preserved."""
  if np.all((0 <= pts) * (pts <= 1)):
    # Points are given in relative coordinates. Convert them.
    pts = pts * np.flip(image.shape[:2])

  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  max_width = max(int(width_a), int(width_b))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  max_height = max(int(heightA), int(heightB))

  dst = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]], dtype='float32')

  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (max_width, max_height))

  return warped

def find_salient_rect(image):
  """Returns the largest contour in the given binarized image, which can be roughly approximated
  by a quadrilateral."""
  contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

  for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approximation) == 4:
      return approximation

def resize_to_fit(image, dim):
  """Resizes the given image to fit the given dimensions, keeping its aspect ratio. The computed
  scaling factor is returned along with the rescaled image."""
  (height, width) = image.shape[:2]
  scale = min(dim[0] / width, dim[1] / height)
  return cv2.resize(image, (round(width * scale), round(height * scale)))

def soften_image(image):
  (height, width) = image.shape[:2]

  sigma = max((2 * (width / 4000) - 1), 0)
  kernel = 5 + 2 * (width // 4200)

  return cv2.GaussianBlur(image, (kernel, kernel), sigma, sigma)

def extract_contours(image, filter_contours):
  """Returns a copy of the given image, containing only the contours (connected components of pixels)
  which match the given filtering function."""
  contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  contours = filter_contours(contours)
  return cv2.drawContours(np.zeros_like(image), contours, -1, (255, 255, 255), thickness=-1)

def mask_components(image, kernel_size, filter_contours):
  """Removes "noise" from the image.

  First, this method performs a morphological dilation of the image with an ellipse kernel of the
  given size. Then it filters the resulting contours (connected components of pixels) using the
  provided filtering function. The result of this operation is applied as a mask to the original
  image and only the masked areas are included in the final result."""
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
  dilated = cv2.dilate(image, kernel)
  mask = extract_contours(dilated, filter_contours)
  return cv2.bitwise_and(mask, image)

def clear_background(image, polygon, bg_color = 255):
  """Clears the background of an image - that is the area which lies outside of the given polygon.

  The polygon must be specified in relative coordinates, i.e. numbers between 0 and 1, representing
  the coordinates as a portion of the image width or height respectively."""
  # Order the corners in clockwise direction, starting from the top-left corner.
  corners = order_points(np.multiply(polygon, np.flip(image.shape[:2])))
  # Find the vectors corresponding to each edge of the polygon. For example, if C1 and C2 are the
  # top-left and top-right corners of the polygon, then the top edge vector is E1 = (C2 - C1).
  edge_vectors = np.roll(corners, -1, axis=0) - corners
  # Create a coordinate matrix which contains the ij coordinats of each pixel of the image in
  # the corresponding position.
  indices = np.transpose(np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij'), (1, 2, 0))
  # Flip the ij indices, so they become xy coordinates.
  indices = np.flip(indices, axis=2)
  # For each pixel, repeat its coordinates four times in preparation to compare them to each of
  # the four corners of the polygon.
  indices = indices.reshape(np.insert(indices.shape, 2, 1)).repeat(4, axis=2)
  # For each pixel P = (x, y), and each polygon corner Ci, find the vector Pi = (P - Ci)
  point_vectors = np.subtract(indices, corners)
  # Find the cross products of Pi and Ei, i = 1,...,4. The sign of the cross products will tell
  # us on which side of each edge lies the corresponding point.
  point_directions = np.cross(point_vectors, edge_vectors)
  # Since the edges were taken in clockwise direction, if a point lies to the left of any of
  # the edges, this means that it lies outside of the polygon.
  outside_polygon = (point_directions > 0).any(axis=2)

  # For all pixels that lie outside of the polygon, set their color to the background color.
  cleared_background = image.copy()
  cleared_background[outside_polygon] = bg_color

  return cleared_background

def shrink_points(grid, max_area=None):
  """Shrinks each contour in the given black & white image to a single point.

  This method finds all contours in the given image and replaces them with their corresponding
  center of mass. If the optional `max_area` parameter is specified, all contours which have
  an area greater than the given value are discarded entirely."""
  contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  if max_area is not None:
    contours = [contour for contour in contours if cv2.contourArea(contour) <= max_area]

  result = np.zeros_like(grid)
  for cluster in contours:
    coords = cluster.reshape((-1, 2)).mean(axis=0).astype(np.int32)
    result[tuple(np.flip(coords))] = 255

  return result

def morph_cross(size, angle):
  """Generates a cross-like kernel to be used for morphological operations.

  The cross can be optionally rotated around its center by the given angle (in radians)."""
  centerX = centerY = size // 2
  a = tan(angle)
  b = centerY - a * centerX
  xs = np.arange(size)
  ys = np.round(a * xs + b).astype(np.int)
  valid = np.logical_and(xs < size, np.logical_and(ys >= 0, ys < size))
  xs = xs[valid]
  ys = ys[valid]
  cross = np.zeros((size, size), dtype=np.uint8)
  cross[ys, xs] = cross[size - xs - 1, ys] = 1
  return cross

def translate(image, translation):
  """Translates the image vertically an horizontally by the given amounts."""
  translation_matrix = np.array([[1, 0, translation[0]], [0, 1, translation[1]]], dtype=np.float32)
  return cv2.warpAffine(image, translation_matrix, np.flip(image.shape))

def find_wavelen_stats(img, wavelen_min, wavelen_max):
  freq_magnitudes = np.abs(np.fft.rfft(img))
  freq_range = [1 / wavelen_max, 1 / wavelen_min]

  frequencies = np.fft.rfftfreq(img.shape[1])
  # Calculate the indices of the frequency range
  freq_in_range = (freq_range[0] < frequencies) & (frequencies < freq_range[1])
  freq_index_range = np.array([
    np.argmax(freq_in_range),
    len(freq_in_range) - np.argmax(freq_in_range[::-1]) - 1,
  ])
  # Calculate the prominent frequencies in each row
  max_coeff_indices = freq_magnitudes[:, slice(*freq_index_range)].argmax(axis=1) + freq_index_range[0]
  prominent_frequencies = frequencies.take(max_coeff_indices)
  prominent_wavelengths = 1 / prominent_frequencies
  return np.median(prominent_wavelengths), prominent_wavelengths.std()

def to_absolute(polygon,img):
  """
    Assuming polygon contains four points in relarive coordinates, 
    convert them to absolute coordinates with respect to the image size
  """
  print("To absolute!")
  return np.array(polygon) * np.flip(img.shape[:2])

