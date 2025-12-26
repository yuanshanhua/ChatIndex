from typing import Any, Type, TypeVar, Union, get_args, get_origin, get_type_hints


T = TypeVar("T")


def load_from_json(typ: Type[T], data: Any) -> T:
    """
    从 JSON 加载数据到指定类型
    """

    # 处理允许互相隐式转换的基本类型
    if typ in (int, float) and isinstance(data, (int, float, bool)):
        # 不允许数字转为 bool, 因为这可能是错误
        try:
            return typ(data)
        except (ValueError, TypeError):
            raise TypeError(f"无法将 {data} 转换为 {typ}")
    # 处理需严格匹配的基本类型
    if typ in (str, bool):
        if not isinstance(data, typ):
            raise TypeError(f"无法将 {data} 转换为 {typ}")
        return data
    # 处理 None
    if typ is None and data is None:
        return None

    # 处理列表类型
    if get_origin(typ) is list:
        if not isinstance(data, list):
            raise TypeError(f"无法将 {type(data)} 转换为 {typ}")
        item_type = get_args(typ)[0]
        return [load_from_json(item_type, item) for item in data]

    # 处理其它复杂类型
    if not isinstance(data, dict):
        raise TypeError(f"无法将 {type(data)} 转换为 {typ}")

    # 处理字典类型
    if get_origin(typ) is dict:
        key_type, val_type = get_args(typ)
        return {load_from_json(key_type, k): load_from_json(val_type, v) for k, v in data.items()}

    # 获取类的所有类型提示
    type_hints = get_type_hints(typ)

    # 创建类实例的参数
    kwargs = {}

    for field_name, field_type in type_hints.items():
        # 如果JSON中没有这个字段，跳过
        if field_name not in data:
            continue

        field_value = data[field_name]

        # 处理嵌套类型
        if field_value is not None:
            field_origin = get_origin(field_type)

            # 处理列表类型
            if field_origin is list:
                item_type = get_args(field_type)[0]
                kwargs[field_name] = [load_from_json(item_type, item) for item in field_value]
            # 处理字典类型
            elif field_origin is dict:
                key_type, val_type = get_args(field_type)
                kwargs[field_name] = {k: load_from_json(val_type, v) for k, v in field_value.items()}
            # 处理Union类型（包括Optional）
            elif field_origin is Union:
                # 获取Union中的类型，排除None类型
                types = [t for t in get_args(field_type) if t is not type(None)]
                # 如果只有一个类型，就直接使用；否则尝试每一个类型
                if len(types) == 1:
                    kwargs[field_name] = load_from_json(types[0], field_value)
                else:
                    # 尝试每一个类型，直到成功
                    for t in types:
                        try:
                            kwargs[field_name] = load_from_json(t, field_value)
                            break
                        except Exception:
                            continue
                    else:
                        raise TypeError(f"无法将 {field_value} 转换为 {field_type}")
            # 处理其他类型（基本类型或自定义类）
            else:
                kwargs[field_name] = load_from_json(field_type, field_value)
        else:
            kwargs[field_name] = None

    # 创建实例
    return typ(**kwargs)
