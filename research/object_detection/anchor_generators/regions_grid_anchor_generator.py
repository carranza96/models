"""

"""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators.grid_anchor_generator import _center_size_bbox_to_corners_bbox
from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.utils import ops


class RegionsGridAnchorGenerator(grid_anchor_generator.GridAnchorGenerator):
    """Generates different grid anchors for vertical regions in an Image"""

    def __init__(self,
                 regions_limits,
                 scales,
                 aspect_ratios,
                 base_anchor_size=None,
                 anchor_stride=None,
                 anchor_offset=None,
                 special_cases=[]):
        """Constructs a RegionsGridAnchorGenerator

        Args:
            regions_limits:
            scales: a list of lists of (float) scales, every lists of scales must have same
                    length (S). Shape is (R, S)
            aspect_ratios: a list of lists of (float) aspect ratios, every set of aspect ratios
                           must have same length (A). Shape is (R, A)
            base_anchor_size: base anchor size as height, width ((length-2 float32 list or tensor, default=[256, 256])
            anchor_stride: difference in centers between base anchors for adjacent
                           grid positions (length-2 float32 list or tensor,
                           default=[16, 16])
            anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                           upper left element of the grid, this should be zero for
                           feature networks with only VALID padding and even receptive
                           field size, but may need additional calculation if other
                           padding is used (length-2 float32 list or tensor,
                           default=[0, 0])
            special_cases: list of special anchors to add to the grid. It is a list of lists of 4 values
                           corresponding to [x_position, y_position, anchor_index, scale, aspect_ratio].
                           x_position is a float (0,1) representing the x position where to add the anchor.
                           y_position is a float (0,1) representing the y position where to add the anchor.
                           anchor_index is an integer representing which anchor to substitute from the given position.
                           scale and aspect_ratio are floats describing the special anchor.
        """
        if not isinstance(regions_limits, list) or not isinstance(scales, list) or not isinstance(aspect_ratios, list):
            raise ValueError("region_limits, scales and aspect_ratios are expected to be lists")
        if not len(scales) == len(aspect_ratios) == (len(regions_limits) + 1):
            raise ValueError("Number of scales and/or aspect_ratio do not correspond to the regions defined.")
        if not all([isinstance(list_item, list) or isinstance(list_item, tuple) for list_item in scales]) or not all(
                [len(list_item) == len(scales[0]) for list_item in scales]):
            raise ValueError("scales is expected to be a list of lists or tuples of same length")
        if not all([isinstance(list_item, list) or isinstance(list_item, tuple) for list_item in
                    aspect_ratios]) or not all(
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
        self._special_cases = special_cases

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
        """Generates a collection of bounding boxes to be used as anchors.

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

        regions = [(0 if i == 0 else self._regions_limits[i - 1],
                    1 if i == len(self._regions_limits) else self._regions_limits[i])
                   for i in range(len(self._regions_limits) + 1)]

        grid_height, grid_width = feature_map_shape_list[0]

        scales_grid, aspect_ratios_grid = [], []
        for region_index in range(len(regions)):
            scales_grid_region, aspect_ratios_grid_region = ops.meshgrid(self._scales[region_index],
                                                                         self._aspect_ratios[region_index])
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
                                       self._anchor_offset,
                                       self._special_cases)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is None:
            num_anchors = anchors.num_boxes()
        anchor_indices = tf.zeros([num_anchors])
        anchors.add_field('feature_map_index', anchor_indices)
        return [anchors]


def tile_anchors_regions(regions,
                         grid_height,
                         grid_width,
                         scales,
                         aspect_ratios,
                         base_anchor_size,
                         anchor_stride,
                         anchor_offset,
                         special_cases):
    """Create a tiled set of anchors strided along a grid in image space.

    Args:
      regions: list of [min, max) values of each vertical regions
      grid_height: size of the grid in the y direction (int or int scalar tensor)
      grid_width: size of the grid in the x direction (int or int scalar tensor)
      scales: a 2-d  (float) tensor representing the scale of each box in the
        basis set for each regions.
      aspect_ratios: a 2-d (float) tensor representing the aspect ratio of each
        box in the basis set for each regions.  The shape of the scales and aspect_ratios tensors
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
      special_cases: list of special anchors to add to the grid. It is a list of lists of 4 values
                     corresponding to [x_position, y_position, anchor_index, scale, aspect_ratio].
                     x_position is a float (0,1) representing the x position where to add the anchor.
                     y_position is a float (0,1) representing the y position where to add the anchor.
                     anchor_index is an integer representing which anchor to substitute from the given position.
                     scale and aspect_ratio are floats describing the special anchor.
    Returns:
      a BoxList holding a collection of N anchor boxes
    """
    x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

    y_centers = tf.cast(tf.range(grid_height), dtype=tf.float32)
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]

    y_centers_regions = [y_centers[
                                    tf.cast(region_min * tf.cast(grid_height, tf.float32), tf.int32):
                                    tf.cast(region_max * tf.cast(grid_height, tf.float32), tf.int32)
                                  ]
                         for region_min, region_max in regions]

    centers = [ops.meshgrid(x_centers, y_centers) for y_centers in y_centers_regions]
    x_centers_region, y_centers_region = [c[0] for c in centers], [c[1] for c in centers]

    ratio_sqrts_region = [tf.sqrt(aspect_ratios[region_index]) for region_index in range(len(regions))]
    heights_region = [scales[region_index] / ratio_sqrts_region[region_index] * base_anchor_size[0]
                      for region_index in range(len(regions))]
    widths_region = [scales[region_index] * ratio_sqrts_region[region_index] * base_anchor_size[1]
                     for region_index in range(len(regions))]

    horizontal_grid = [ops.meshgrid(widths_region[region_index], x_centers_region[region_index]) for region_index in
                       range(len(regions))]
    vertical_grid = [ops.meshgrid(heights_region[region_index], y_centers_region[region_index]) for region_index in
                     range(len(regions))]

    widths_grid, x_centers_grid = tf.concat([x[0] for x in horizontal_grid], axis=0), tf.concat(
        [x[1] for x in horizontal_grid], axis=0)
    heights_grid, y_centers_grid = tf.concat([x[0] for x in vertical_grid], axis=0), tf.concat(
        [x[1] for x in vertical_grid], axis=0)

    special_cases_index = []
    special_cases_weights = []
    special_cases_heights = []
    for x_position, y_position, anchor_index, scale, aspect_ratio in special_cases:
        x_index = tf.cast(x_position * tf.cast(grid_width, tf.float32), tf.int32)
        y_index = tf.cast(y_position * tf.cast(grid_height, tf.float32), tf.int32)
        sqrt_ar = np.sqrt(aspect_ratio)
        special_cases_weights.append(scale / sqrt_ar * base_anchor_size[0])
        special_cases_heights.append(scale * sqrt_ar * base_anchor_size[1])
        special_cases_index.append([y_index, x_index, tf.cast(anchor_index, tf.int32)])
    widths_grid = tf.tensor_scatter_nd_update(widths_grid, [[special_cases_index]], [[special_cases_weights]])
    heights_grid = tf.tensor_scatter_nd_update(heights_grid, [[special_cases_index]], [[special_cases_heights]])

    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)

    return box_list.BoxList(bbox_corners)
