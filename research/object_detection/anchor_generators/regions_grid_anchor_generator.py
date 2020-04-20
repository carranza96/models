"""

"""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops
from object_detection.core import box_list
from object_detection.utils import ops


class RegionsGridAnchorGenerator(anchor_generator.AnchorGenerator):
    """ """

    def __init__(self,
                 regions_limits,
                 scales,
                 aspect_ratios,
                 base_anchor_size=None,
                 anchor_stride=None,
                 anchor_offset=None):
        """Constructs a RegionsGridAnchorGenerator

        Args:
            regions_limits:
            scales:
            aspect_ratios:
            base_anchor_size:
            anchor_stride:
            anchor_offset:
        """
        if not isinstance(regions_limits, list) or not isinstance(scales, list) or not isinstance(aspect_ratios, list):
            raise ValueError("region_limits, scales and aspect_ratios are expected to be lists")
        if not len(scales) == len(aspect_ratios) == (len(regions_limits)+1):
            raise ValueError("Number of scales and/or aspect_ratio do not correspond to the regions defined.")
        if not all([isinstance(list_item, list) or isinstance(list_item, tuple) for list_item in scales]) or not all(
                [len(list_item) == len(scales[0]) for list_item in scales]):
            raise ValueError("scales is expected to be a list of lists or tuples of same length")
        if not all([isinstance(list_item, list) or isinstance(list_item, tuple) for list_item in aspect_ratios]) or not all(
                [len(list_item) == len(scales[0]) for list_item in aspect_ratios]):
            raise ValueError("scales is expected to be a list of lists or tuples of same length")
        if not all([0 < list_item < 1 for list_item in regions_limits]):
            raise ValueError("regions_limits are expected to be in the range (0, 1)")

        # Handle argument defaults
        if base_anchor_size is None:
            base_anchor_size = [256, 256]
        if anchor_stride is None:
            anchor_stride = [16, 16]
        if anchor_offset is None:
            anchor_offset = [0, 0]

        self._regions_limits = regions_limits
        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
        self._anchor_stride = anchor_stride
        self._anchor_offset = anchor_offset

    def name_scope(self):
        return 'RegionsGridAnchorGenerator'

    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location.

        Returns:
          a list of integers, one for each expected feature map to be passed to
          the `generate` function.
        """
        return [len(self._scales[0]) * len(self._aspect_ratios[0])]

    def _generate(self, feature_map_shape_list):
        """TODO:Generates a collection of bounding boxes to be used as anchors.

        Args:
          feature_map_shape_list: list of pairs of convnet layer resolutions in the
            format [(height_0, width_0)].  For example, setting
            feature_map_shape_list=[(8, 8)] asks for anchors that correspond
            to an 8x8 layer.  For this anchor generator, only lists of length 1 are
            allowed.

        Returns:
          boxes_list: a list of BoxLists each holding anchor boxes corresponding to
            the input feature map shapes.

        Raises:
          ValueError: if feature_map_shape_list, box_specs_list do not have the same
            length.
          ValueError: if feature_map_shape_list does not consist of pairs of
            integers
        """
        if not (isinstance(feature_map_shape_list, list)
                and len(feature_map_shape_list) == 1):
            raise ValueError('feature_map_shape_list must be a list of length 1.')
        if not all([isinstance(list_item, tuple) and len(list_item) == 2
                    for list_item in feature_map_shape_list]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')

        # Create constants in init_scope so they can be created in tf.functions
        # and accessed from outside of the function.
        with tf.init_scope():
            self._base_anchor_size = tf.cast(tf.convert_to_tensor(
                self._base_anchor_size), dtype=tf.float32)
            self._anchor_stride = tf.cast(tf.convert_to_tensor(
                self._anchor_stride), dtype=tf.float32)
            self._anchor_offset = tf.cast(tf.convert_to_tensor(
                self._anchor_offset), dtype=tf.float32)

        regions = [(0 if i == 0 else self._regions_limits[i-1],
                    1 if i == len(self._regions_limits) else self._regions_limits[i])
                   for i in range(len(self._regions_limits)+1)]

        grid_height, grid_width = feature_map_shape_list[0]

        scales_grid, aspect_ratios_grid = [], []
        for region_index in range(len(regions)):
            scales_grid_region, aspect_ratios_grid_region = ops.meshgrid(self._scales[region_index], self._aspect_ratios[region_index])
            scales_grid_region = tf.reshape(scales_grid_region, [-1])
            aspect_ratios_grid_region = tf.reshape(aspect_ratios_grid_region, [-1])
            scales_grid.append(scales_grid_region)
            aspect_ratios_grid.append(aspect_ratios_grid_region)

        anchors = tile_anchors_regions(regions,
                                       grid_height,
                                       grid_width,
                                       scales_grid,
                                       aspect_ratios_grid,
                                       self._base_anchor_size,
                                       self._anchor_stride,
                                       self._anchor_offset)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is None:
            num_anchors = anchors.num_boxes()
        anchor_indices = tf.zeros([num_anchors])
        anchors.add_field('feature_map_index', anchor_indices)
        print(anchors)
        return [anchors]

def tile_anchors_regions(regions,
                         grid_height,
                         grid_width,
                         scales,
                         aspect_ratios,
                         base_anchor_size,
                         anchor_stride,
                         anchor_offset):
    """TODO:Create a tiled set of anchors strided along a grid in image space.

    This op creates a set of anchor boxes by placing a "basis" collection of
    boxes with user-specified scales and aspect ratios centered at evenly
    distributed points along a grid.  The basis collection is specified via the
    scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
    and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
    .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
    and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
    placing it over its respective center.

    Grid points are specified via grid_height, grid_width parameters as well as
    the anchor_stride and anchor_offset parameters.

    Args:
      grid_height: size of the grid in the y direction (int or int scalar tensor)
      grid_width: size of the grid in the x direction (int or int scalar tensor)
      scales: a 1-d  (float) tensor representing the scale of each box in the
        basis set.
      aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
        box in the basis set.  The length of the scales and aspect_ratios tensors
        must be equal.
      base_anchor_size: base anchor size as [height, width]
        (float tensor of shape [2])
      anchor_stride: difference in centers between base anchors for adjacent grid
                     positions (float tensor of shape [2])
      anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need some additional calculation if other
                     padding is used (float tensor of shape [2])
    Returns:
      a BoxList holding a collection of N anchor boxes
    """
    bbox_corners = tf.constant([], shape=(0,4), dtype=tf.float32)

    x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

    for region_index, (region_min, region_max) in enumerate(regions):
        ratio_sqrts = tf.sqrt(aspect_ratios[region_index])
        heights = scales[region_index] / ratio_sqrts * base_anchor_size[0]
        widths = scales[region_index] * ratio_sqrts * base_anchor_size[1]

        # Get a grid of box centers
        region_y_offset = anchor_offset[0] if region_index==0 else anchor_stride[0] - int(region_min * grid_height) % anchor_stride[0]
        y_centers = tf.cast(tf.range(int(region_min * grid_height), int(region_max * grid_height)), dtype=tf.float32)
        y_centers = y_centers * anchor_stride[0] + anchor_offset[0] + region_y_offset
        x_centers_region, y_centers_region = ops.meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers_region)
        heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers_region)

        bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = tf.reshape(bbox_centers, [-1, 2])
        bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
        bbox_corners = tf.concat([bbox_corners, _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)], axis=0)
    return box_list.BoxList(bbox_corners)

def _center_size_bbox_to_corners_bbox(centers, sizes):
    """Converts bbox center-size representation to corners representation.

    Args:
      centers: a tensor with shape [N, 2] representing bounding box centers
      sizes: a tensor with shape [N, 2] representing bounding boxes

    Returns:
      corners: tensor with shape [N, 4] representing bounding boxes in corners
        representation
    """
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
