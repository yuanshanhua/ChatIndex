from transformers import PreTrainedModel, PreTrainedTokenizer

from index_advisor.ia_logging import logger
from index_advisor.mm.model import COLUMN_TOKEN, SQL_TOKEN, MMConfig, MyModel
from llamafactory.model.model_utils.visual import _register_composite_model


logger = logger.getChild("hook.model")
_config = MMConfig()


def mm_init(cfg: MMConfig):
    global _config
    _config = cfg


def config() -> MMConfig:
    return _config


def hook_tokenizer(tokenizer: PreTrainedTokenizer):
    logger.debug(f"Hooking tokenizer: {tokenizer.__class__.__name__}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"添加 pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    # 添加 <column> token
    if _config.enable:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=[COLUMN_TOKEN]),  # type: ignore
            replace_additional_special_tokens=False,
        )
        _config.column_token_id = tokenizer.get_added_vocab()[COLUMN_TOKEN]
        logger.info(f"添加 <column> token: {COLUMN_TOKEN} (ID: {_config.column_token_id})")
    if _config.enable_sql:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=[SQL_TOKEN]),  # type: ignore
            replace_additional_special_tokens=False,
        )
        _config.sql_token_id = tokenizer.get_added_vocab()[SQL_TOKEN]
        logger.info(f"添加 <sql> token: {SQL_TOKEN} (ID: {_config.sql_token_id})")
    return tokenizer


def hook_load_model(model: "PreTrainedModel"):
    logger.debug(f"Hooking model: {model.__class__.__name__}")
    if not _config.enable and not _config.enable_sql:
        return model
    wrapped_model = MyModel(model, _config)
    return wrapped_model


def hook_rl_model(model: "PreTrainedModel"):
    if not _config.enable and not _config.enable_sql:
        return model

    if not _config.lm_trainable:
        _config.lm_trainable = True
        logger.warning("⚠️ RL 不允许冻结语言模型, 此参数无意义")

    if not _config.projector_trainable and not _config.projector2_trainable:
        _config.projector_trainable = _config.enable
        _config.projector2_trainable = _config.enable_sql
        logger.warning(
            f"⚠️ RL 不允许冻结多模态投影层参数, 已启用 {_config.enable * 'column'} {_config.enable_sql * 'sql'} 投影层训练"
        )

    return _set_and_check_params(model)


def hook_sft_model(model: "PreTrainedModel"):
    # if not _config.enable and not _config.enable_sql:
    #     raise ValueError("未启用多模态但进行 SFT, 非预期行为")

    # 确保至少有一部分参数是可训练的
    if not _config.lm_trainable and not _config.projector_trainable and not _config.projector2_trainable:
        raise ValueError("语言模型和多模态投影层都被冻结，无法进行训练")

    return _set_and_check_params(model)


def _set_and_check_params(model: "PreTrainedModel") -> "PreTrainedModel":
    logger.info(f"{'✅' if _config.lm_trainable else '❌'}训练语言模型")
    logger.info(f"{'✅' if _config.projector_trainable else '❌'}训练 column 投影层")
    logger.info(f"{'✅' if _config.projector2_trainable else '❌'}训练 sql 投影层")

    # 设置模型的训练模式
    model.train()

    lora = False
    if any("lora" in k.lower() for k in model.state_dict().keys()):
        logger.info("检测到 LoRA adapter, 将仅设置 adapter 为可训练")
        lora = True
        assert _config.lm_trainable, "使用 LoRA adapter 时 lm_trainable 必须为 True"

    # 统计参数数量
    trainable_params = 0
    total_params = 0
    projector_params = 0
    projector2_params = 0
    lm_params = 0
    adapter_params = 0

    # 遍历所有参数，根据配置设置是否可训练
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "multi_modal_projector2" in name:
            projector2_params += param.numel()
            param.requires_grad = _config.projector2_trainable
        elif "multi_modal_projector" in name:
            # 多模态投影层参数
            projector_params += param.numel()
            param.requires_grad = _config.projector_trainable
        elif "lora" in name.lower():
            # lora adapter
            adapter_params += param.numel()
            param.requires_grad = _config.lm_trainable and param.requires_grad
        else:
            lm_params += param.numel()
            # 当 lm_trainable 为 False 时，冻结语言模型参数, 否则维持原状.
            param.requires_grad = _config.lm_trainable and param.requires_grad and not lora

        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"可训练: {name}")
        else:
            logger.debug(f"冻结: {name}")

    # 打印参数统计信息
    logger.info("参数统计:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  语言模型参数: {lm_params:,}")
    if adapter_params:
        logger.info(f"  LoRA adapter 参数: {adapter_params:,}")
    logger.info(f"  column 投影层参数: {projector_params:,}")
    logger.info(f"  sql 投影层参数: {projector2_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")

    # 验证至少有一些参数是可训练的
    if trainable_params == 0:
        raise ValueError("没有可训练的参数！请检查配置。")

    return model


_register_composite_model("qwen_table")  # 避免 multi_modal_projector 被添加 lora
