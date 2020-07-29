from __future__ import print_function

import copy
import six
import warnings

import functools
from paddle.fluid import layers
from paddle.fluid import framework
from paddle.fluid import core
from paddle.fluid import name_scope
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.clip import GradientClipBase, _correct_clip_op_role_var

class DoubleClip(GradientClipBase):
    def __init__(self, clip_value, clip_norm, group_name="default_group", need_clip=None):
        super(DoubleClip, self).__init__(need_clip)
        self.clip_value = float(clip_value)
        self.clip_norm = float(clip_norm)
        self.group_name = group_name

    def __str__(self):
        return "Gradient Clip By Value and GlobalNorm, value={}, global_norm={}".format(
            self.clip_value, self.clip_norm)

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_grads = self._dygraph_clip_by_value(params_grads)
        params_grads = self._dygraph_clip_by_global_norm(params_grads)
        return params_grads

    @imperative_base.no_grad
    def _dygraph_clip_by_value(self, params_grads):
        params_and_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = layers.clip(x=g, min=-self.clip_value, max=self.clip_value)
            params_and_grads.append((p, new_grad))
        return params_and_grads
    
    @imperative_base.no_grad
    def _dygraph_clip_by_global_norm(self, params_grads):
        params_and_grads = []
        sum_square_list = []
        for p, g in params_grads:
            if g is None:
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.merge_selected_rows(g)
                merge_grad = layers.get_tensor_from_selected_rows(merge_grad)
            square = layers.square(merge_grad)
            sum_square = layers.reduce_sum(square)
            sum_square_list.append(sum_square)

        # all parameters have been filterd out
        if len(sum_square_list) == 0:
            return params_grads

        global_norm_var = layers.concat(sum_square_list)
        global_norm_var = layers.reduce_sum(global_norm_var)
        global_norm_var = layers.sqrt(global_norm_var)
        max_global_norm = layers.fill_constant(
            shape=[1], dtype='float32', value=self.clip_norm)
        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
        for p, g in params_grads:
            if g is None:
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = layers.elementwise_mul(x=g, y=clip_var)
            params_and_grads.append((p, new_grad))

        return params_and_grads