# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet V1 Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import resnet_utils
from nets import resnet_v1
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops

slim = contrib_slim


class FasterRCNNResnetV1FpnFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Resnet V1 feature extractor implementation."""

  def __init__(self,
               architecture,
               resnet_model,
               is_training,
               first_stage_features_stride,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               activation_fn=tf.nn.relu,
               fpn_scope_name='fpn',
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False
  ):
    """Constructor.

    Args:
      architecture: Architecture name of the Resnet V1 model.
      resnet_model: Definition of the Resnet V1 model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      activation_fn: Activaton functon to use in Resnet V1 model.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._architecture = architecture
    self._resnet_model = resnet_model
    self._activation_fn = activation_fn
    super(FasterRCNNResnetV1FpnFeatureExtractor,
          self).__init__(is_training, first_stage_features_stride,
                         batch_norm_trainable, reuse_weights, weight_decay)
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._pad_to_multiple = pad_to_multiple
    self._fpn_scope_name = fpn_scope_name
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._additional_layer_depth = additional_layer_depth
    self._use_native_resize_op = use_native_resize_op
    self._override_base_feature_extractor_hyperparams = override_base_feature_extractor_hyperparams


  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      return resized_inputs - [[channel_means]]

    elif resized_inputs.shape.as_list()[3] == 6:
      channel_means = [123.68, 116.779, 103.939, 131.979, 26.662, 12.622]
      return resized_inputs - [[channel_means]]

    else:
      return resized_inputs


  def _filter_features(self, image_features):
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block1', 'block2', 'block3', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features


  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              activation_fn=self._activation_fn,
              weight_decay=self._weight_decay)):
        with (slim.arg_scope(self._conv_hyperparams_fn()) if
        self._override_base_feature_extractor_hyperparams else context_manager.IdentityContextManager()):
            with tf.variable_scope(self._architecture, reuse=self._reuse_weights) as var_scope:
                with (slim.arg_scope(self._conv_hyperparams_fn())
                if self._override_base_feature_extractor_hyperparams else
                context_manager.IdentityContextManager()):
                  _, image_features = self._resnet_model(
                      inputs=ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                      num_classes=None,
                      is_training=self._train_batch_norm,
                      global_pool=False,
                      output_stride=None,
                      spatial_squeeze=False,
                      store_non_strided_activations=True,
                      scope=var_scope)
                  image_features = self._filter_features(image_features)

    depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)

    with slim.arg_scope(self._conv_hyperparams_fn()):
          with tf.variable_scope(self._fpn_scope_name, reuse=self._reuse_weights):
              base_fpn_max_level = min(self._fpn_max_level, 5)
              feature_block_list = []
              for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                  feature_block_list.append('block{}'.format(level - 1))

              fpn_features = feature_map_generators.fpn_top_down_feature_maps(
                  [(key, image_features[key]) for key in feature_block_list],
                  depth=depth_fn(self._additional_layer_depth),
                  use_native_resize_op=self._use_native_resize_op)

              feature_maps = []
              for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                  feature_maps.append(
                      fpn_features['top_down_block{}'.format(level - 1)])
              last_feature_map = fpn_features['top_down_block{}'.format(
                  base_fpn_max_level - 1)]
              # Construct coarse features
              for i in range(base_fpn_max_level, self._fpn_max_level):
                  last_feature_map = slim.conv2d(
                      last_feature_map,
                      num_outputs=depth_fn(self._additional_layer_depth),
                      kernel_size=[3, 3],
                      stride=2,
                      padding='SAME',
                      scope='bottom_up_block{}'.format(i))
                  feature_maps.append(last_feature_map)

    return feature_maps, None


  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    # with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
    #   with slim.arg_scope(
    #       resnet_utils.resnet_arg_scope(
    #           batch_norm_epsilon=1e-5,
    #           batch_norm_scale=True,
    #           activation_fn=self._activation_fn,
    #           weight_decay=self._weight_decay)):
    #     with slim.arg_scope([slim.batch_norm],
    #                         is_training=self._train_batch_norm):
    #       blocks = [
    #           resnet_utils.Block('block4', resnet_v1.bottleneck, [{
    #               'depth': 2048,
    #               'depth_bottleneck': 512,
    #               'stride': 1
    #           }] * 3)
    #       ]
    #       proposal_classifier_features = resnet_utils.stack_blocks_dense(
    #           proposal_feature_maps, blocks)
    return proposal_feature_maps


class FasterRCNNResnet50FpnFeatureExtractor(FasterRCNNResnetV1FpnFeatureExtractor):
  """Faster R-CNN Resnet 50 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               activation_fn=tf.nn.relu,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False
  ):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      activation_fn: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet50FpnFeatureExtractor,
          self).__init__('resnet_v1_50',
                         resnet_v1.resnet_v1_50,
                         is_training,
                         first_stage_features_stride,
                         depth_multiplier,
                         min_depth,
                         conv_hyperparams_fn,
                         pad_to_multiple,
                         batch_norm_trainable,
                         reuse_weights,
                         weight_decay,
                         activation_fn,
                         'fpn',
                         fpn_min_level,
                         fpn_max_level,
                         additional_layer_depth,
                         use_native_resize_op,
                         override_base_feature_extractor_hyperparams
                         )


class FasterRCNNResnet101FpnFeatureExtractor(FasterRCNNResnetV1FpnFeatureExtractor):
  """Faster R-CNN Resnet 101 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               activation_fn=tf.nn.relu,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False
               ):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      activation_fn: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet101FpnFeatureExtractor,
          self).__init__('resnet_v1_101',
                         resnet_v1.resnet_v1_101,
                         is_training,
                         first_stage_features_stride,
                         depth_multiplier,
                         min_depth,
                         conv_hyperparams_fn,
                         pad_to_multiple,
                         batch_norm_trainable,
                         reuse_weights,
                         weight_decay,
                         activation_fn,
                         'fpn',
                         fpn_min_level,
                         fpn_max_level,
                         additional_layer_depth,
                         use_native_resize_op,
                         override_base_feature_extractor_hyperparams
                         )


class FasterRCNNResnet152FpnFeatureExtractor(FasterRCNNResnetV1FpnFeatureExtractor):
  """Faster R-CNN Resnet 152 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               depth_multiplier,
               min_depth,
               conv_hyperparams_fn,
               pad_to_multiple,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               activation_fn=tf.nn.relu,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False
               ):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      activation_fn: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNResnet152FpnFeatureExtractor,
          self).__init__('resnet_v1_152',
                         resnet_v1.resnet_v1_152,
                         is_training,
                         first_stage_features_stride,
                         depth_multiplier,
                         min_depth,
                         conv_hyperparams_fn,
                         pad_to_multiple,
                         batch_norm_trainable,
                         reuse_weights,
                         weight_decay,
                         activation_fn,
                         'fpn',
                         fpn_min_level,
                         fpn_max_level,
                         additional_layer_depth,
                         use_native_resize_op,
                         override_base_feature_extractor_hyperparams
                         )
