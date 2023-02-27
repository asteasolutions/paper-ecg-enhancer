from functools import cached_property
import os
import cv2
import numpy as np
from scipy.signal import find_peaks
from math import ceil, pi, tan
from pipetools import pipe
from .pipe_cache import pipe_cache
from .utils import extract_contours, find_wavelen_stats, four_point_transform, mask_components, normalize_multi_channel, normalize_channel, adjust_levels, resize_to_fit

PROCESSING_IMAGE_SIZE = 4000
ESTIMATED_GRID_GAP = 15
GRID_GAP_LOWER_BOUND = ESTIMATED_GRID_GAP / 2
GRID_GAP_UPPER_BOUND = ESTIMATED_GRID_GAP * 2

WINDOW_REL_HEIGHT = 0.04
WINDOW_REL_WIDTH = 0.04
WINDOW_OVERLAP = 0.9
REL_PADDING = (WINDOW_REL_HEIGHT, WINDOW_REL_WIDTH)
MIN_COUNTOURS_PER_WINDOW = 3

class Windowing:
  def __init__(self, img, rel_width, rel_height, rel_overlap):
    self.__img = img
    self.window_shape = np.ceil(np.multiply([rel_height, rel_width], img.shape[:2])).astype(int)
    self.overlap = ceil(self.window_shape[1] * rel_overlap)

  @property
  def window_height(self):
    return self.window_shape[0]

  @property
  def window_width(self):
    return self.window_shape[1]

  @cached_property
  def step(self):
    return self.window_width - self.overlap

  @cached_property
  def shape(self):
    return (
      ceil(self.__img.shape[0] / self.window_height),
      ceil((self.__img.shape[1] - self.window_width) / self.step) + 1
    )

  def __getitem__(self, key):
    i, j = key
    if not 0 <= i < self.shape[0] or not 0 <= j < self.shape[1]:
      raise IndexError()

    return (
      slice(i * self.window_height, (i + 1) * self.window_height),
      slice(j * self.step, j * self.step + self.window_width)
    )

  def center_point(self, i, j):
    y = i * self.window_height + self.window_height // 2
    x = j * self.step + self.window_width // 2
    return (y, x)

  def __str__(self):
    return (
      '[Windowing shape: %s, image_shape: %s, window_shape: %s, overlap: %s]'
      % (self.shape, self.__img.shape, self.window_shape, self.overlap)
    )


def approximate_grid_gaps(image, range):
  """
  Gives three approximations of the grid gaps in the image. The first number approximates
  the horizontal grid gap, the second one approximates the vertical grid gap and the third
  is an overall approximation - an average of the previous two. These numbers should not
  be treated as exact values, because they are just averages and their corresponding
  standard deviations could be significant, e.g. if the paper is skewed or curled.
  """
  horiz_gap, _ = find_wavelen_stats(image, range[0], range[1])
  vert_gap, _ = find_wavelen_stats(np.rot90(image), range[0], range[1])

  return round(horiz_gap), round(vert_gap), round(np.mean([horiz_gap, vert_gap]))

def round_to_odd(num):
  return (num // 2) * 2 + 1

def build_horizontal_margin_eraser_kernel(vertical_margin, max_angle):
  """
  The horizontal margin eraser kernel is a convolution kernel that erases points which
  lie in some margin around a horizontal line. Put diferently, after applying this kernel
  the output image will contain only points that lie on isolated horizontal lines.

  vertical_margin: the size of the margin around the horizontal line that needs to be clear
  max_angle: the maximum deviation angle from the X-axis for which a line is considered "horizontal"

  The output kernel will look something like this (with width and height adjusted
  to match the given parameters):

  -1 -1 -1 -1 -1
  -1 -1 -1 -1 -1
   0  0 -1  0  0
   0  0  1  0  0
   0  0 -1  0  0
  -1 -1 -1 -1 -1
  -1 -1 -1 -1 -1

  When applied to a binarized uint8 image, each pixel will become white only if it is already
  white _and_ none of the pixels matched by any of the -1s in the kernel is white.
  """
  kernel_height = 2 * vertical_margin + 1
  # Set a lower bound on the `max_angle` parameter, because if it is zero,
  # its tangent will also be zero and the kernel will become infinitely wide.
  tangent = tan(max(max_angle, pi / 16))
  # 3 is the height of a zero-filled stripe that will fill the areas to the left and right
  # of the target pixel. By calculating the width of the kernel this way, we ensure that
  # iff the horizontal line's deviation angle from the X-axis is less than `max_angle`,
  # the whole line will stay inside this zero-filled stripe.
  vertical_margin = int(3 / tangent)
  kernel_width = 2 * vertical_margin + 1

  kernel = np.zeros((kernel_height, kernel_width), dtype=int)
  # Set the central pixel to 1
  kernel[kernel_height // 2, kernel_width // 2] = 1
  # Set the pixels immediately above and below of the central pixel to -1
  kernel[kernel_height // 2 - 1, kernel_width // 2] = -1
  kernel[kernel_height // 2 + 1, kernel_width // 2] = -1
  # Fill the whole top and bottom margins with -1s
  kernel[:(kernel_height // 2 - 1), :] = -1
  kernel[(kernel_height // 2 + 2):, :] = -1

  return kernel

def calc_straight_grid(img, processing_grid):
  cell_size = np.ceil(np.divide(img.shape[:2], processing_grid)).astype(int)
  grid = np.zeros((*np.add(processing_grid, (2,)), 2), dtype=int)
  ys = np.round([0, *np.linspace(cell_size[1] // 2, img.shape[0] - cell_size[1] // 2, processing_grid[0]), img.shape[0] - 1]).astype(int)
  xs = np.round([0, *np.linspace(cell_size[0] // 2, img.shape[1] - cell_size[0] // 2, processing_grid[1]), img.shape[1] - 1]).astype(int)
  for i, y in enumerate(ys):
    for j, x in enumerate(xs):
      grid[i, j] = [y, x]

  return grid

def calc_curved_grid(horiz_lines, vert_lines, processing_grid, padding):
  crosspoints = np.zeros((*np.add(processing_grid, (2,)), 2), dtype=int)
  cy = 0
  for i in range(horiz_lines.shape[0]):
    if horiz_lines[i, 0] == 0:
      continue

    y = i
    cx = 0
    for x in range(horiz_lines.shape[1]):
      if vert_lines[y, x] > 0:
        coords = np.subtract((y, x), padding)
        if cx == 0 or abs(crosspoints[cy, cx - 1, 1] - coords[1]) > 1:
          crosspoints[cy, cx] = coords
          cx += 1
      if x < horiz_lines.shape[1] - 1:
        if y > 0 and horiz_lines[y - 1, x + 1] > 0:
          y -= 1
        elif y < horiz_lines.shape[0] - 1 and horiz_lines[y + 1, x + 1] > 0:
          y += 1

    cy += 1

  return crosspoints

def calc_horiz_gradient(lines):
  # Extract all lines as contours
  contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  # Filter only lines that are long at least 1% of the image width
  contours = [c for c in contours if cv2.boundingRect(c)[2] > 0.1 * lines.shape[1]]

  # For each line calculate the change in y per unit change in x
  y_deltas = []
  for contour in contours:
    vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    if abs(vy) > abs(vx) / 2:
      # Line is not horizontal. Ignore.
      continue

    y_deltas.append(vy / vx)

  return np.median(y_deltas) if len(y_deltas) >= MIN_COUNTOURS_PER_WINDOW else np.nan

def detect_lines(img, approx_grid_gap, windowing, padding, horizontal):
  if horizontal:
    return detect_horiz_lines(img, approx_grid_gap, windowing, padding)

  # To detect vertical lines, rotate the image 90 degrees, detect horizontal lines,
  # and then rotate the result 90 degress in the opposite direction.
  return np.rot90(
    detect_horiz_lines(np.rot90(img), approx_grid_gap, windowing, np.flip(padding)),
    axes=(1, 0)
  )

def interpolate_curve(curve):
  if np.sum(np.isnan(curve)) > curve.shape[0] / 2:
    # Can't interpolate if it's mostly NaN's
    return

  for i in range(curve.shape[0]):
    if np.isnan(curve[i]):
      prev = None
      prev_delta_i = None
      prev_points = curve[:i][::-1]
      if np.any(np.isfinite(prev_points)):
        prev_delta_i = np.argmax(np.isfinite(prev_points)) + 1
        prev = curve[i - prev_delta_i]

      next = None
      next_delta_i = None
      next_points = curve[(i+1):]
      if np.any(np.isfinite(next_points)):
        next_delta_i = np.argmax(np.isfinite(next_points)) + 1
        next = curve[i + next_delta_i]

      if prev is None:
        curve[i] = next
      elif next is None:
        curve[i] = prev
      else:
        curve[i] = (next_delta_i * prev + prev_delta_i * next) / (prev_delta_i + next_delta_i)

def calc_gradients(img, windowing):
  gradients = np.zeros(windowing.shape, dtype=np.float64)
  for i in range(windowing.shape[0]):
    for j in range(windowing.shape[1]):
      gradients[i, j] = calc_horiz_gradient(img[windowing[i, j]])

    interpolate_curve(gradients[i, :])

  for i in range(gradients.shape[0] // 2, -1, -1):
    if np.any(np.isnan(gradients[i, :])):
      gradients[i, :] = gradients[i + 1, :]

  for i in range(gradients.shape[0] // 2, gradients.shape[0]):
    if np.any(np.isnan(gradients[i, :])):
      gradients[i, :] = gradients[i - 1, :]

  if np.any(np.isnan(gradients)):
    raise Exception(
      'Can\'t interpolate gradients with shape %s. Missing gradients at positions %s.'
      % (gradients.shape, np.argwhere(np.isnan(gradients)))
    )

  return gradients

def detect_horiz_lines(img, approx_grid_gap, windowing, padding):
  blurred = cv2.blur(img, ksize=(round_to_odd(approx_grid_gap), 3))
  kernel = np.array([
    [-2, -3, -4, -3, -2],
    [ 1,  1,  1,  1,  1],
    [ 2,  4,  6,  4,  2],
    [ 1,  1,  1,  1,  1],
    [-2, -3, -4, -3, -2],
  ])
  gradients = cv2.filter2D(blurred, -1, kernel)
  gradients_normalized = cv2.adaptiveThreshold(
    gradients,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=round_to_odd(5 * approx_grid_gap),
    C=0,
  )
  bool_lines = np.bool8(gradients_normalized)
  bool_lines = np.logical_and(bool_lines, np.logical_not(np.roll(bool_lines, axis=0, shift=1)))
  horiz_lines = bool_lines.astype(np.uint8) * 255
  salient_lines = extract_contours(horiz_lines, lambda cntrs: [c for c in cntrs if cv2.boundingRect(c)[2] > 10 * approx_grid_gap])
  kernel = build_horizontal_margin_eraser_kernel(round(0.75 * approx_grid_gap), pi / 8)
  lines = cv2.filter2D(salient_lines, -1, kernel)

  # Dilate the lines horizontally to fill small gaps
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round_to_odd(approx_grid_gap), 1))
  lines = cv2.dilate(lines, kernel)

  lines = extract_contours(lines, lambda cntrs: [c for c in cntrs if cv2.boundingRect(c)[2] > 20 * approx_grid_gap])

  gradients = calc_gradients(lines, windowing)
  curvature_lines = np.pad(np.zeros_like(lines), np.reshape(padding, (-1, 1)))

  def draw_line(ys):
    for x, y in enumerate(ys):
      if np.any(np.greater_equal([round(y), x], curvature_lines.shape[:2])) or np.any(
        np.less([round(y), x], [0, 0])
      ):
        continue

      curvature_lines[round(y), x] = 255

  for i, row in enumerate(gradients):
    center_y, center_x = np.add(windowing.center_point(i, len(row) // 2), padding)
    y_deltas = [row[0]] * (padding[1] + windowing.window_width // 2)
    for j in range(len(row) - 1):
      y_deltas.extend(np.linspace(row[j], row[j + 1], windowing.step, endpoint=False))
    y_deltas.extend([row[-1]] * (padding[1] + windowing.window_width // 2))

    y_deltas.insert(center_x, 0)
    ys = np.add(center_y, [*-np.cumsum(y_deltas[:center_x][::-1])[::-1], *np.cumsum(y_deltas[center_x:])])

    draw_line(ys)
    if i == 0:
      draw_line(np.subtract(ys, windowing.window_height // 2))

    if i == len(gradients) - 1:
      draw_line(np.add(ys, windowing.window_height // 2))

  return curvature_lines

def map_grid(image, src, dst):
  mapped = np.ones_like(image) * 255
  for i in range(src.shape[0] - 1):
    for j in range(src.shape[1] - 1):
      src_rect = np.flip([src[i, j], src[i, j + 1], src[i + 1, j + 1], src[i + 1, j]], axis=1).astype(np.float32)
      width = dst[i, j + 1, 1] - dst[i, j, 1]
      height = dst[i + 1, j, 0] - dst[i, j, 0]
      dst_rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
      M = cv2.getPerspectiveTransform(src_rect, dst_rect)
      mapped[dst[i, j, 0]:dst[i + 1, j, 0], dst[i, j, 1]:dst[i, j + 1, 1]] = cv2.warpPerspective(
        image, M, (width, height), borderMode=cv2.BORDER_REPLICATE
      )

  return mapped

def mask_signal(image, estimated_grid_gap):
  salient_contour_area = max(20, image.shape[0] * image.shape[1] // 10000)
  return mask_components(
    image,
    kernel_size=estimated_grid_gap,
    filter_contours=lambda cntrs: [c for c in cntrs if cv2.contourArea(c) > salient_contour_area]
  )

def adjust_horiz_grid_gaps(image, vert_line_locations, target_gap):
  height = image.shape[0]
  stripes = [image[:, :vert_line_locations[0]]]
  target_stripe_rect = np.array(
    [[0, 0], [target_gap - 1, 0], [target_gap - 1, height - 1], [0, height - 1]],
    dtype=np.float32
  )
  gaps = vert_line_locations[1:] - vert_line_locations[:-1]
  for i, gap in enumerate(gaps):
    x1 = vert_line_locations[i]
    x2 = vert_line_locations[i + 1]
    if 0.8 * target_gap < gap < 1.2 * target_gap and gap != target_gap:
      src_rect = np.array(
        [[x1, 0], [x2 - 1, 0], [x2 - 1, height - 1], [x1, height - 1]],
        dtype=np.float32
      )
      M = cv2.getPerspectiveTransform(src_rect, target_stripe_rect)
      stripe = cv2.warpPerspective(image, M, (target_gap, height), borderMode=cv2.BORDER_REPLICATE)
      stripes.append(stripe)
    else:
      stripes.append(image[:, x1:x2])

  stripes.append(image[:, vert_line_locations[-1]:])
  return np.concatenate(stripes, axis=1)

def process_ecg(path, ecg_polygon):
  file_name = os.path.splitext(os.path.basename(path))[0]
  print('Processing %s...' % file_name)

  log = bool(os.getenv('DEBUG'))
  K, load, _, tap = pipe_cache(log_prefix=file_name, debug_logs=log)

  return (path > pipe
    | _(cv2.imread)
    # Preform a perspective transformation to bring up only the ECG paper
    | _(four_point_transform, pts=np.array(ecg_polygon))
    # Resize the ECG to fit inside a predefined size.
    | _(resize_to_fit, dim=(PROCESSING_IMAGE_SIZE, 99999), K='ecg')
    # Create a grayscale version for computation purposes
    | _(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY, K='grayscale_ecg')
    # Calculate an approximation of the gap between ECG grid lines, i.e. 1mm.
    | tap(approximate_grid_gaps, range=(GRID_GAP_LOWER_BOUND, GRID_GAP_UPPER_BOUND), K='grid_gaps')
    # Get an odd version of the grid gap approximation to be used as morph kernel size.
    | tap(lambda _: round_to_odd(load('grid_gaps')[2]), K='grid_gap_odd')
    # Normalize the lighting of the grayscale ECG
    | _(normalize_channel, ksize=(K('grid_gap_odd'),) * 2)
    # Invert the colors so that we have white lines on black background
    | _(cv2.bitwise_not)
    # The ECG will be processed in regions specified by the windowing parameters.
    # First, calculate some padding to be added around the image where the continuation
    # of the curved lines will be drawn. This will help us straighten the border regions
    # of the image correclty.
    | tap(lambda img: np.ceil(np.multiply(img.shape[:2], REL_PADDING)).astype(int), K='padding')
    # Then, separately detect the horizontal and vertical lines in the image.
    | tap(Windowing, rel_width=WINDOW_REL_WIDTH, rel_height=WINDOW_REL_HEIGHT, rel_overlap=WINDOW_OVERLAP, K='horiz_windowing')
    | tap(detect_lines, approx_grid_gap=K('grid_gaps')[0], windowing=K('horiz_windowing'), padding=K('padding'), horizontal=True, K='horiz_lines')
    | tap(lambda img: Windowing(np.rot90(img), rel_width=WINDOW_REL_WIDTH, rel_height=WINDOW_REL_HEIGHT, rel_overlap=WINDOW_OVERLAP), K='vert_windowing')
    | tap(detect_lines, approx_grid_gap=K('grid_gaps')[1], windowing=K('vert_windowing'), padding=K('padding'), horizontal=False, K='vert_lines')
    | tap(lambda _: (load('vert_windowing').shape[0], load('horiz_windowing').shape[0]), K='processing_grid')
    # Trace the horizontal lines and find their intersections with the vertical lines, thus
    # calculating the crossing points of a "curved" grid, describing the curvature of the ECG paper.
    | tap(lambda _: calc_curved_grid(load('horiz_lines'), load('vert_lines'), load('processing_grid'), load('padding')), K='curved_grid')
    # Calculate the "straight" grid. It is just a rectangular breakdown of the image.
    | tap(calc_straight_grid, processing_grid=K('processing_grid'), K='straight_grid')
    # Normalize the lighting of the original (color) ECG image.
    | _(lambda _: normalize_multi_channel(load('ecg'), ksize=(load('grid_gap_odd'),) * 2))
    | _(adjust_levels, bin_size=1, low_percentile=0.01, high_percentile=0.99)
    # Straighten the ECG paper by mapping the curved grid to the straight one.
    | _(map_grid, src=K('curved_grid'), dst=K('straight_grid'), K='straight_ecg')

    # Additional adjustments: rescale each time step so that they are all equal
    # First, convert the ECG to grayscale and erase the signal so that we are left only with the grid.
    | _(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
    | _(normalize_channel, ksize = (K('grid_gap_odd'),) * 2, K='normalized')
    | _(adjust_levels, bin_size=1, low_percentile=0.001, high_percentile=0.1)
    | _(cv2.threshold, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    | _(lambda x: x[1])
    | _(mask_signal, estimated_grid_gap=K('grid_gap_odd'))
    | _(cv2.bitwise_or, K('normalized'))
    # Make the grid white on black
    | _(cv2.threshold, thresh=0, maxval=255, type=cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    | _(lambda x: x[1])
    # Compute a histogram by counting the white pixels in each column. We expect vertical
    # grid lines to contain the largest number of white pixels
    | _(np.count_nonzero, axis=0)
    # Find the peaks in the histogram
    | _(find_peaks, distance=K('grid_gaps')[0] * 0.75, prominence=300, wlen=K('grid_gaps')[0] * 2)
    # For each two adjacent peaks that are "almost" one grid gap apart, stretch or shrink
    # the corresponding stripe of the image so that it becomes *exactly* one grid gap wide.
    | _(lambda peaks: adjust_horiz_grid_gaps(load('straight_ecg'), peaks[0], load('grid_gaps')[0]))
  )
