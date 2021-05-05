import os
import argparse

#
# main function
#
#
#

# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', nargs='+', type=str, help='The input file(s)')

# Get the arguments
args = parser.parse_args()

# Process the test image(s)
for input_file in args.input:
  # Determine the file type
  filename, extension = os.path.splitext(input_file)

  if extension == ".jpg":
    print("Processing Image " + input_file)
    #handle_image(input_file, args.output, matrix, distortion)
  elif extension == ".png":
    print("Processing Image " + input_file)
    #handle_image(input_file, args.output, matrix, distortion)
  elif extension == ".mp4":
    print("Processing Video " + input_file)
    #handle_video(input_file, matrix, distortion)
  else:
    print("Unknown extension: " + extension)