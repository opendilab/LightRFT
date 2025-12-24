lightrft.strategy.fsdp.fsdp_utils
=================================================

.. currentmodule:: lightrft.strategy.fsdp.fsdp_utils

.. automodule:: lightrft.strategy.fsdp.fsdp_utils



is_meta_initialized
----------------------------------------------------------

.. autofunction:: is_meta_initialized



multi_tensor_l2norm_torch
----------------------------------------------------------

.. autofunction:: multi_tensor_l2norm_torch



calc_l2_norm
----------------------------------------------------------

.. autofunction:: calc_l2_norm



calc_lp
----------------------------------------------------------

.. autofunction:: calc_lp



get_norm
----------------------------------------------------------

.. autofunction:: get_norm



reduce_grads
----------------------------------------------------------

.. autofunction:: reduce_grads



get_tensor_norm
----------------------------------------------------------

.. autofunction:: get_tensor_norm



compute_norm
----------------------------------------------------------

.. autofunction:: compute_norm



BaseGradScaler
----------------------------------------------------------

.. autoclass:: BaseGradScaler
    :members: scale, inv_scale, state_dict, load_state_dict, update



DynamicGradScaler
----------------------------------------------------------

.. autoclass:: DynamicGradScaler
    :members: update, state_dict, load_state_dict



BaseOptimizer
----------------------------------------------------------

.. autoclass:: BaseOptimizer
    :members: param_groups, defaults, add_param_group, step, zero_grad, load_state_dict, state_dict, backward, backward_by_grad, clip_grad_norm


