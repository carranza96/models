""""ATSS (Adaptive Training Sample Selector) matcher implementation.

This class takes a similarity matrix and matches columns to rows based on the
maximum value per column. One can specify matched_thresholds and
to prevent columns from matching to rows (generally resulting in a negative
training example) and unmatched_theshold to ignore the match (generally
resulting in neither a positive or negative training example).

This matcher is used in Fast(er)-RCNN.


"""
import tensorflow as tf
import abc
import six
from numpy import number
from object_detection.utils import shape_utils
from object_detection.core.matcher import Match
from object_detection.core import box_list



class CenterMatcher(six.with_metaclass(abc.ABCMeta, object)):
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  matched_threshold (upper threshold) and unmatched_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= matched_threshold: Highest similarity. Matched/Positive!
  (2) matched_threshold > similarity >= unmatched_threshold: Medium similarity.
          Depending on negatives_lower_than_unmatched, this is either
          Unmatched/Negative OR Ignore.
  (3) unmatched_threshold > similarity: Lowest similarity. Depending on flag
          negatives_lower_than_unmatched, either Unmatched/Negative OR Ignore.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(self,
               max_assignments=12,
               force_match_for_each_row=False,
               use_matmul_gather=False,
               ):
    """Construct ATSSMatcher.

    """
    self._max_assignments = max_assignments
    self._force_match_for_each_row = force_match_for_each_row
    self._use_matmul_gather = use_matmul_gather


  def match(self, groundtruth_boxes, anchors, similarity_matrix, valid_rows=None, scope=None):
    """Computes matches among row and column indices and returns the result.

    Computes matches among the row and column indices based on the similarity
    matrix and optional arguments.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher value means more similar.
      valid_rows: A boolean tensor of shape [N] indicating the rows that are
        valid for matching.
      scope: Op scope name. Defaults to 'Match' if None.

    Returns:
      A Match object with the results of matching.
    """
    with tf.name_scope(scope, 'Match') as scope:
      if valid_rows is None:
        valid_rows = tf.ones(tf.shape(similarity_matrix)[0], dtype=tf.bool)
      return Match(self._match(groundtruth_boxes, anchors, similarity_matrix, valid_rows),
                   self._use_matmul_gather)

  def _match(self, groundtruth_boxes, anchors, similarity_matrix, valid_rows):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.
      valid_rows: a boolean tensor of shape [N] indicating valid rows.

    Returns:
      Match object with corresponding matches for each of M columns.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
          similarity_matrix)
      return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """

      selected_anchors_by_center_in_area = anchor_centers_inside_gt(groundtruth_boxes, anchors)

      ## Version 1
      # top_k_anchors_per_gt = tf.math.top_k(tf.cast(selected_anchors_by_center_in_area,tf.float32) * similarity_matrix, k=30)[1]
      #
      # iou_selected_anchors = tf.gather(similarity_matrix, top_k_anchors_per_gt, axis=1, batch_dims=1)
      #
      # mean_iou_selected_anchors = tf.reduce_mean(iou_selected_anchors, axis=1)
      # std_iou_selected_anchors = tf.math.reduce_std(iou_selected_anchors, axis=1)
      #
      # iou_thresholds = mean_iou_selected_anchors + std_iou_selected_anchors

      ## Version 2

      num_selected_anchors_per_gt = tf.cast(tf.math.count_nonzero(selected_anchors_by_center_in_area, axis=1), tf.float32)

      iou_selected_anchors = tf.cast(selected_anchors_by_center_in_area,tf.float32) * similarity_matrix
      mean_iou_selected_anchors = tf.math.divide_no_nan(tf.reduce_sum(iou_selected_anchors, axis=1), num_selected_anchors_per_gt)

      substract_mean = (iou_selected_anchors - tf.expand_dims(mean_iou_selected_anchors, axis=1)) \
                       * tf.cast(selected_anchors_by_center_in_area, tf.float32)

      std_iou_selected_anchors = tf.math.sqrt(tf.math.divide_no_nan(
          tf.reduce_sum(tf.pow(substract_mean, 2), axis=1),
          num_selected_anchors_per_gt))

      iou_thresholds = mean_iou_selected_anchors + 2*std_iou_selected_anchors

      invalid_thresholds = tf.where(tf.equal(iou_thresholds,0))

      iou_thresholds = tf.tensor_scatter_nd_update(iou_thresholds, tf.cast(invalid_thresholds, tf.int32),
                                  tf.ones(tf.shape(invalid_thresholds)[0]))



      selected_anchors_by_threshold = tf.cast(tf.greater_equal(similarity_matrix, tf.expand_dims(iou_thresholds, 1)), tf.int32)

      selected_anchors = selected_anchors_by_center_in_area * selected_anchors_by_threshold

      iou_values_positive_anchors = tf.cast(selected_anchors, tf.float32) * similarity_matrix

      #
      # # # Limit to top max_assignments anchors
      top_k_anchors_per_gt = tf.math.top_k(iou_values_positive_anchors, k=self._max_assignments)[1]

      # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
      range_rows = tf.expand_dims(tf.range(0, tf.shape(top_k_anchors_per_gt)[0]), 1)  # will be [[0], [1]]
      range_rows_repeated = tf.tile(range_rows, [1, tf.shape(top_k_anchors_per_gt)[1]])  # will be [[0, 0], [1, 1]]
      # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
      full_indices = tf.concat([tf.expand_dims(range_rows_repeated, -1), tf.expand_dims(top_k_anchors_per_gt, -1)],
                               axis=2)
      full_indices = tf.reshape(full_indices, [-1, 2])

      selected_anchors_by_max_assignments = tf.cast(
          tf.scatter_nd(full_indices, tf.ones(tf.size(top_k_anchors_per_gt)), tf.shape(similarity_matrix)), tf.float32)

      selected_anchors = tf.cast(selected_anchors, tf.float32) * selected_anchors_by_max_assignments


      iou_values_positive_anchors = tf.cast(selected_anchors, tf.float32) * similarity_matrix

      mask_negative_anchors = tf.equal(tf.reduce_sum(selected_anchors, axis=0), 0)

      matches = tf.argmax(iou_values_positive_anchors, 0, output_type=tf.int32)

      matches = self._set_values_using_indicator(matches, mask_negative_anchors, -1)

      matches = tf.reshape(matches,[tf.shape(anchors.get())[0]])


      # TODO: Force Match for each row??
      # TODO: Should we ignore any anchor as in ArgMaxMatcher??
      # TODO: Any differences between proposal and detection stages??


      if self._force_match_for_each_row:
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(
              similarity_matrix)
        force_match_column_ids = tf.argmax(similarity_matrix, 1,
                                             output_type=tf.int32)
        force_match_column_indicators = (
              tf.one_hot(
                  force_match_column_ids, depth=similarity_matrix_shape[1]) *
              tf.cast(tf.expand_dims(valid_rows, axis=-1), dtype=tf.float32))
        force_match_row_ids = tf.argmax(force_match_column_indicators, 0,
                                          output_type=tf.int32)
        force_match_column_mask = tf.cast(
              tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        final_matches = tf.where(force_match_column_mask,
                                   force_match_row_ids, matches)
        return final_matches
      else:
        return matches


    if similarity_matrix.shape.is_fully_defined():
      if shape_utils.get_dim_as_int(similarity_matrix.shape[0]) == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:
      return tf.cond(
          tf.greater(tf.shape(similarity_matrix)[0], 0),
          _match_when_rows_are_non_empty, _match_when_rows_are_empty)


  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)


def anchor_centers_inside_gt(groundtruth_boxes, anchors):
    ycenter2, xcenter2, _, _ = box_list.BoxList.get_center_coordinates_and_sizes(anchors)

    # [y_min, x_min, y_max, x_max]
    gt_boxes_tensor = groundtruth_boxes.get()
    gt_boxes_broadcast_ymin = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 0), (tf.shape(gt_boxes_tensor)[0], 1)))
    gt_boxes_broadcast_xmin = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 1), (tf.shape(gt_boxes_tensor)[0], 1)))
    gt_boxes_broadcast_ymax = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 2), (tf.shape(gt_boxes_tensor)[0], 1)))
    gt_boxes_broadcast_xmax = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 3), (tf.shape(gt_boxes_tensor)[0], 1)))

    is_in_xmin = tf.greater_equal(xcenter2 - tf.transpose([gt_boxes_broadcast_xmin]), 0)
    is_in_ymin = tf.greater_equal(ycenter2 - tf.transpose([gt_boxes_broadcast_ymin]), 0)
    is_in_xmax = tf.less_equal(xcenter2 - tf.transpose([gt_boxes_broadcast_xmax]), 0)
    is_in_ymax = tf.less_equal(ycenter2 - tf.transpose([gt_boxes_broadcast_ymax]), 0)
    selected_anchors_by_center_in_area = tf.logical_and(tf.logical_and(is_in_xmin, is_in_ymin),
                                                        tf.logical_and(is_in_xmax, is_in_ymax))

    # Mask similarly to selected_anchors_by_threshold or selected_anchors_by_distance
    selected_anchors_by_center_in_area = tf.cast(selected_anchors_by_center_in_area, tf.int32)
    selected_anchors_by_center_in_area = \
        tf.cond(tf.equal(tf.size(tf.shape(selected_anchors_by_center_in_area)),1),
            lambda: tf.expand_dims(selected_anchors_by_center_in_area,axis=0),
            lambda: selected_anchors_by_center_in_area)
    return selected_anchors_by_center_in_area




## Ragged tensor version
# num_selected_anchors_per_gt = tf.expand_dims(
#     tf.cast(tf.math.count_nonzero(selected_anchors_by_center_in_area, axis=1), tf.float32), axis=1)
# indices = tf.where(selected_anchors_by_center_in_area)
# ragged_iou_selected_anchors = tf.RaggedTensor.from_value_rowids(
#     values=tf.gather_nd(similarity_matrix,indices),
#     value_rowids=indices[...,0])
#
# mean_iou_selected_anchors = tf.reduce_mean(ragged_iou_selected_anchors, axis=1)
# # iou_selected_anchors = tf.cast(selected_anchors_by_center_in_area, tf.float32) * similarity_matrix
# # mean_iou_selected_anchors =  tf.cond(tf.equal(tf.size(tf.shape(mean_iou_selected_anchors)),1),
# #       lambda: tf.expand_dims(mean_iou_selected_anchors,axis=1),
# #       lambda: mean_iou_selected_anchors)
# # substract_mean = ragged_iou_selected_anchors - tf.expand_dims(mean_iou_selected_anchors,axis=1)
# substract_mean = tf.math.add(ragged_iou_selected_anchors, -tf.expand_dims(mean_iou_selected_anchors,axis=1))
# std_iou_selected_anchors = tf.math.sqrt(tf.reduce_sum(tf.pow(substract_mean, 2) / num_selected_anchors_per_gt, axis=1))
#
# iou_thresholds = mean_iou_selected_anchors + 2*std_iou_selected_anchors