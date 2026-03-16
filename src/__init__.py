from verl.utils.device import is_npu_available

if is_npu_available:
    from verl.models.transformers import npu_patch as npu_patch