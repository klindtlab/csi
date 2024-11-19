#!/bin/bash

# Exit on error
set -e

echo "Starting ImageNetX download script..."

# Create directory if it doesn't exist
mkdir -p imagenetx

wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget -P imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

git clone https://github.com/facebookresearch/imagenetx.git

# Replace np.bool with bool (for general usage)
find . -type f -name "*.py" -exec sed -i 's/np.bool/bool/g' {} +

echo "Download complete!"


