import glob
import os
import sys
import shutil
from PIL import Image
import traceback
import io
import gzip

# We download stare_images.tar and labels_ah.tar
# from the following website.
#
# https://cecas.clemson.edu/~ahoover/stare/probing/index.html

def extract_masks(input_dir,  output_dir):
    ppm_gz_files = glob.glob(input_dir + "/*.ppm.gz")
    for ppm_gz_file in ppm_gz_files:
        with gzip.open(ppm_gz_file, 'rb') as f_in:
            file_content = f_in.read()

            image = Image.open(io.BytesIO(file_content))
            w, h  = image.size
            # Expand image.
            image = image.resize((w*5, h*5))
            basename = os.path.basename(ppm_gz_file)
            name     = basename.split(".")[0]
            output_file = os.path.join(output_dir, name + ".jpg")
            image.save(output_file, 'JPEG')
            print("Saved {}".format(output_file))

def extract_images(input_dir,  output_dir, output_masks_dir):
    ppm_gz_files = glob.glob(input_dir + "/*.ppm.gz")
    for ppm_gz_file in ppm_gz_files:
        basename = os.path.basename(ppm_gz_file)
        name     = basename.split(".")[0]
        mask_filepath = os.path.join(output_masks_dir, name + ".jpg")
        if not os.path.exists(mask_filepath):
          continue

        with gzip.open(ppm_gz_file, 'rb') as f_in:
           
            file_content = f_in.read()

            image = Image.open(io.BytesIO(file_content))
            w, h  = image.size
            # Expand image.
            image = image.resize((w*5, h*5))
            output_file = os.path.join(output_dir, name + ".jpg")
            image.save(output_file, 'JPEG')
            print("Saved {}".format(output_file))

if __name__ == "__main__":
  try:
    labels_dir       = "./labels-ah/"
    output_masks_dir = "./STARE-master/masks"
    if os.path.exists(output_masks_dir):
        shutil.rmtree(output_masks_dir)
    os.makedirs(output_masks_dir)

    extract_masks(labels_dir, output_masks_dir)

    images_dir  = "./stare-images/"
    output_images_dir  = "./STARE-master/images"
    if os.path.exists(output_images_dir):
        shutil.rmtree(output_images_dir)
    os.makedirs(output_images_dir)
    extract_images(images_dir, output_images_dir, output_masks_dir )
  except:

    traceback.print_exc()
