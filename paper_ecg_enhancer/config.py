import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
OUTPUT_DIR = os.getenv('OUTPUT_DIR') or os.path.join(ROOT_DIR, 'out')

# Ensure output dir exists
if not os.path.isdir(OUTPUT_DIR):
  os.mkdir(OUTPUT_DIR)
