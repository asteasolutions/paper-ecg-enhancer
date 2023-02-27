import os
import sys
from paper_ecg_enhancer.process_ecg import process_ecg

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('\nUsage:\n\tpython run.py <image_path> <polygon>\n\n' +
      'Where polygon is specified in relative x,y coordinates separated by semicolons, e.g. "0.2,0.3;0.8,0.2;0.8,0.8;0.2,0.7"')
    exit(1)

  _, image_path, polygon_str = sys.argv[:3]

  points = polygon_str.split(';')
  polygon = [list(map(float, point.split(','))) for point in points]

  os.environ['DEBUG'] = 'true'
  process_ecg(image_path, polygon)
