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
    """
    :alias_main: paddle.nn.GradientClipByGlobalNorm
	:alias: paddle.nn.GradientClipByGlobalNorm,paddle.nn.clip.GradientClipByGlobalNorm
	:old_api: paddle.fluid.clip.GradientClipByGlobalNorm

    Given a list of Tensor :math:`t\_list` , calculate the global norm for the elements of all tensors in 
    :math:`t\_list` , and limit it to ``clip_norm`` .
    
    - If the global norm is greater than ``clip_norm`` , all elements of :math:`t\_list` will be compressed by a ratio.
    
    - If the global norm is less than or equal to ``clip_norm`` , nothing will be done.
    
    The list of Tensor :math:`t\_list` is not passed from this class, but the gradients of all parameters in ``Program`` . If ``need_clip``
    is not None, then only part of gradients can be selected for gradient clipping.
    
    Gradient clip will takes effect after being set in ``optimizer`` , see the document ``optimizer`` 
    (for example: :ref:`api_fluid_optimizer_SGDOptimizer`).

    The clipping formula is:

    .. math::

        t\_list[i] = t\_list[i] * \\frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    Args:
        clip_norm (float): The maximum norm value.
        group_name (str, optional): The group name for this clip. Default value is ``default_group``
        need_clip (function, optional): Type: function. This function accepts a ``Parameter`` and returns ``bool`` 
            (True: the gradient of this ``Parameter`` need to be clipped, False: not need). Default: None, 
            and gradients of all parameters in the network will be clipped.

    Examples:
        .. code-block:: python
        
            # use for Static mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np
                        
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(
                    main_program=main_prog, startup_program=startup_prog):
                image = fluid.data(
                    name='x', shape=[-1, 2], dtype='float32')
                predict = fluid.layers.fc(input=image, size=3, act='relu') # Trainable parameters: fc_0.w.0, fc_0.b.0
                loss = fluid.layers.mean(predict)
                
                # Clip all parameters in network:
                clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
                
                # Clip a part of parameters in network: (e.g. fc_0.w_0)
                # pass a function(fileter_func) to need_clip, and fileter_func receive a ParamBase, and return bool
                # def fileter_func(Parameter):
                # # It can be easily filtered by Parameter.name (name can be set in fluid.ParamAttr, and the default name is fc_0.w_0, fc_0.b_0)
                #   return Parameter.name=="fc_0.w_0"
                # clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0, need_clip=fileter_func)

                sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1, grad_clip=clip)
                sgd_optimizer.minimize(loss)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            x = np.random.uniform(-100, 100, (10, 2)).astype('float32')
            exe.run(startup_prog)
            out = exe.run(main_prog, feed={'x': x}, fetch_list=loss)


            # use for Dygraph mode
            import paddle
            import paddle.fluid as fluid

            with fluid.dygraph.guard():
                linear = fluid.dygraph.Linear(10, 10)  # Trainable: linear_0.w.0, linear_0.b.0
                inputs = fluid.layers.uniform_random([32, 10]).astype('float32')
                out = linear(fluid.dygraph.to_variable(inputs))
                loss = fluid.layers.reduce_mean(out)
                loss.backward()

                # Clip all parameters in network:
                clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)

                # Clip a part of parameters in network: (e.g. linear_0.w_0)
                # pass a function(fileter_func) to need_clip, and fileter_func receive a ParamBase, and return bool
                # def fileter_func(ParamBase):
                # # It can be easily filtered by ParamBase.name(name can be set in fluid.ParamAttr, and the default name is linear_0.w_0, linear_0.b_0)
                #   return ParamBase.name == "linear_0.w_0"
                # # Note: linear.weight and linear.bias can return the weight and bias of dygraph.Linear, respectively, and can be used to filter
                #   return ParamBase.name == linear.weight.name
                # clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0, need_clip=fileter_func)

                sgd_optimizer = fluid.optimizer.SGD(
                    learning_rate=0.1, parameter_list=linear.parameters(), grad_clip=clip)
                sgd_optimizer.minimize(loss)

    """

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
        params_and_grads = []
        # clip by value first
        for p, g in params_grads:
            if g is None:
                continue
            if self._need_clip_func is not None and not self._need_clip_func(p):
                params_and_grads.append((p, g))
                continue
            new_grad = layers.clip(x=g, min=-self.clip_value, max=self.clip_value)
            params_and_grads.append((p, new_grad))
        params_grads = params_and_grads
        
        # clip by global norm
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
