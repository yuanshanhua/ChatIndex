import os
from collections.abc import MutableMapping

from transformers import is_torch_npu_available

from llamafactory.extras.misc import get_device_count

from .utils import parse_ranges


def set_env_devices(env: MutableMapping, devices: list | str | int):
    """
    通过环境变量设置可见设备. 支持 GPU 和 Ascend NPU.
    """
    if isinstance(devices, list):
        devices = ",".join(map(str, devices))
    elif isinstance(devices, int):
        devices = str(devices)
    if is_torch_npu_available():
        env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    else:
        env["CUDA_VISIBLE_DEVICES"] = devices


def set_device(s: str | None):
    """
    从参数或环境变量中解析设备 ID 列表.
    参数字符串需使用逗号分隔或 start:end:step 格式.
    注意: 此函数会修改 os.environ 以影响子进程, 对当前进程无效.
    """
    if s is not None:
        device_ids = parse_ranges(s)
        set_env_devices(os.environ, device_ids)
    elif os.getenv("CUDA_VISIBLE_DEVICES"):
        device_ids = list(map(int, (os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))))
    elif os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        device_ids = list(map(int, (os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").split(","))))
    else:
        device_ids = list(range(get_device_count()))
    return device_ids
