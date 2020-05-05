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
from object_detection.utils import shape_utils
from object_detection.core.matcher import Match



class ATSSMatcher(six.with_metaclass(abc.ABCMeta, object)):
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
               k=9,
               use_matmul_gather=False,
               ):
    """Construct ATSSMatcher.

    """
    self._k = k
    self._use_matmul_gather = use_matmul_gather


  def match(self, similarity_matrix, distance_matrix, valid_rows=None, scope=None):
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
      return Match(self._match(similarity_matrix, distance_matrix, valid_rows),
                   self._use_matmul_gather)

  def _match(self, similarity_matrix, distance_matrix, valid_rows):
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
      top_k_anchors_per_gt = tf.math.top_k(distance_matrix, k=self._k)[1]

      iou_selected_anchors = tf.gather(similarity_matrix, top_k_anchors_per_gt, axis=1, batch_dims=1)

      mean_iou_selected_anchors = tf.reduce_mean(iou_selected_anchors, axis=1)
      std_iou_selected_anchors = tf.math.reduce_std(iou_selected_anchors, axis=1)

      iou_thresholds = mean_iou_selected_anchors + std_iou_selected_anchors


      # Remove not selected anchors based on distance
      # top_k_anchor indices will be [[0, 1], [1, 2]]
      # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
      range_rows = tf.expand_dims(tf.range(0, tf.shape(top_k_anchors_per_gt)[0]), 1)  # will be [[0], [1]]
      range_rows_repeated = tf.tile(range_rows, [1, tf.shape(top_k_anchors_per_gt)[1]])  # will be [[0, 0], [1, 1]]
      # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
      full_indices = tf.concat([tf.expand_dims(range_rows_repeated, -1), tf.expand_dims(top_k_anchors_per_gt, -1)],
                               axis=2)
      full_indices = tf.reshape(full_indices, [-1, 2])

      selected_anchors_by_distance = tf.cast(
          tf.scatter_nd(full_indices, tf.ones(tf.size(top_k_anchors_per_gt)), tf.shape(similarity_matrix)), tf.int32)

      selected_anchors_by_threshold = tf.cast(tf.greater_equal(similarity_matrix, tf.expand_dims(iou_thresholds, 1)), tf.int32)

      selected_anchors = selected_anchors_by_distance * selected_anchors_by_threshold

      iou_values_positive_anchors = tf.cast(selected_anchors, tf.float32) * similarity_matrix

      mask_negative_anchors = tf.equal(tf.reduce_sum(selected_anchors, axis=0), 0)

      matches = tf.argmax(iou_values_positive_anchors, 0, output_type=tf.int32)

      matches = self._set_values_using_indicator(matches, mask_negative_anchors, -1)

      # TODO: Remove anchors whose center are not inside the GT box
      # TODO: Force Match for each row??
      # TODO: Should we ignore any anchor as in ArgMaxMatcher??
      # TODO: Any differences between proposal and detection stages??

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
