lightrft.strategy.strategy_base
=================================================

.. currentmodule:: lightrft.strategy.strategy_base

.. automodule:: lightrft.strategy.strategy_base


EngineStatus
----------------------------------------------------------

.. autoenum:: EngineStatus
    :members: SLEEPED,WAKEUP



StrategyBase
----------------------------------------------------------

.. autoclass:: StrategyBase
    :members: __init__,set_seed,setup_distributed,create_optimizer,prepare,backward,optimizer_step,setup_dataloader,save_ckpt,load_ckpt,all_reduce,all_gather,print,is_rank_0,get_rank,unwrap_model,prepare_models_and_optimizers,report_memory,setup_inference_engine,maybe_sleep_inference_engine,wakeup_inference_engine,engine_generate_local,_build_multimodal_inputs,gather_and_generate,update_engine_weights,sync_and_clear_cache,init_model_context,maybe_offload_optimizer,maybe_load_optimizer



is_actor
----------------------------------------------------------

.. autofunction:: is_actor


