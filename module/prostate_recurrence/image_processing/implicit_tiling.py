"""# a set of functions for tiling of whole slide image without saving

Each scene of the whole slide image is tiled according to a user-defined tile
size. The tissue mask of the scene, at the same resolution, is used to filter
out tiles with too low tissue. The tumour mask of the scene, at the same resolution,
is used to annotate the tumour area with a tile.

Implicit tiling is used for extracting tile level features without saving tile
images and needs to be integrated into a step of a feature extraction pipeline.

## read_image(...) to read whole slide image using czifile library.
This function expects the full absolute path to a .czi image image and a valid
tiling method as input parameters. Implemented reading methods include 'czifile' 
and 'aicsimageio'.

## read_image_mask() ...

## create_tiles(...) to create image tiles based on the whole slide image.
This function expects a numpy array of the whole slide image and a user-defined
tile size as input parameters. By default, the tile size is set to be 512 pixels.
This function returns a dictionary of image tiles, each stored as a numpy array, and
the number of rows and columns of the tiled whole slide image.

"""