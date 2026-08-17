"""活体记忆系统 - API 配置

从环境变量读取配置，提供 get_api_config() 返回可直接传给 LivingMemoryLoop
的配置字典。预训练模型默认使用本地缓存路径，避免重复下载。

本模块同时导出共享的配置构建辅助函数，供 mcp_memory_server.py 复用，
消除两处重复的 embedder / LLM API 配置构建逻辑：
  - build_embedder(): 构建 PretrainedEmbedder（失败降级 SimpleEmbedder）
  - build_llm_api_config(): 从环境变量构建 LLM API 配置字典
  - DEFAULT_SYSTEM_PROMPT: 共享的 system prompt 常量

环境变量（按优先级覆盖默认值）:
    DEEPSEEK_API_KEY      -- DeepSeek API 密钥（若提供则启用 LLM）
    LMS_LLM_BASE_URL      -- LLM API 基础 URL（默认 https://api.deepseek.com/v1）
    LMS_LLM_API_KEY       -- LLM API 密钥（与 DEEPSEEK_API_KEY 等效回退）
    LMS_LLM_MODEL         -- LLM 模型名（默认 deepseek-chat）
    LMS_EMBEDDER          -- 嵌入器类型: 'cloud' | 'pretrained' | 'simple'（默认 pretrained）
    LMS_CLOUD_EMBED_URL   -- 云端 embed 服务 URL（LMS_EMBEDDER=cloud 时生效）
    LMS_CLOUD_EMBED_MODEL -- 云端 embed 模型名（如 bge-m3）
    LMS_CLOUD_EMBED_DIM   -- 云端 embed 输出维度（如 1024）
    LMS_CLOUD_EMBED_FALLBACK_URL -- 备用 embed URL（主 URL 失败自动切换；本机 LAN→隧道）
    LMS_SELF_REF_ENABLED  -- 自指回路开关（默认 false；测试期开启，每日复核是否关闭）
    LMS_INPUT_DIM         -- 输入维度（默认 64）
    LMS_NUM_NODES         -- 节点数（默认 256）
    LMS_PRETRAINED_MODEL  -- 预训练模型本地路径（覆盖默认缓存路径）
    LMS_J_TARGET_NORM     -- J 范数钳制目标（allostatic 开启时 = 滑动设定点初值）
    LMS_J_ALLOSTATIC      -- allostatic J 滑动设定点开关（默认 0=关，固定 J 不变）
    LMS_J_ALLOSTATIC_WINDOW / _K / _STEP / _PERSIST / _MIN / _MAX /
    LMS_J_ALLOSTATIC_MIN_SAMPLES / _SAT_FRAC / _COL_ACT
                          -- allostatic 机制参数（详见 runtime/allostatic_j.py）
    LMS_API_HOST          -- 服务监听地址（默认 127.0.0.1，run.py 使用）
    LMS_API_PORT          -- 服务监听端口（默认 8190，run.py 使用）
"""

import os
import logging

from runtime.config import default_config
from core.paths import resolve_pretrained_model_path, get_snapshot_dir

logger = logging.getLogger(__name__)

# 预训练模型路径：优先环境变量 LMS_PRETRAINED_MODEL，自动探测跨平台缓存目录
DEFAULT_PRETRAINED_MODEL_PATH = resolve_pretrained_model_path()

# 默认 LLM 配置：DeepSeek API
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-chat"

# 共享的 system prompt（api/config.py 与 mcp_memory_server.py 复用）
DEFAULT_SYSTEM_PROMPT = (
    '你是一个有记忆能力的AI助手。系统会为你提供海马体的记忆context，'
    '其中包含激活节点信息和长时记忆检索结果，'
    '这代表了你对此前对话的记忆状态。'
    '请结合记忆context和当前用户输入来回答问题。'
    '如果记忆context中有相关信息，请尝试利用它来回答；'
    '同时保持自然流畅的对话。'
)

# 默认服务配置
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8190


def _get_env(key: str, default: str = "") -> str:
    """读取环境变量，去首尾空白。"""
    val = os.environ.get(key, default)
    return val.strip() if isinstance(val, str) else val


def build_embedder(embedder_type: str, input_dim: int):
    """根据类型构造嵌入器实例。

    被 api/config.py 的 get_api_config() 与 mcp_memory_server.py 的
    _build_config() 共享：支持 cloud / pretrained / simple 三种类型，
    依赖缺失或加载失败时自动降级。

    参数:
        embedder_type: 'cloud'、'pretrained' 或 'simple'。
        input_dim: 感官向量维度。

    返回:
        嵌入器实例；若 cloud/pretrained 依赖缺失则自动降级为 simple。
    """
    embedder_type = embedder_type.lower().strip()

    if embedder_type == 'cloud':
        # 使用云端 embed 服务（如手机 Ollama bge-m3）
        try:
            from core.sensory.cloud_embedder import CloudEmbedder
            api_url = _get_env(
                "LMS_CLOUD_EMBED_URL",
                "https://embed.example.com/v1/embeddings")
            model = _get_env("LMS_CLOUD_EMBED_MODEL", "bge-m3")
            remote_dim = int(_get_env("LMS_CLOUD_EMBED_DIM", "1024"))
            logger.info(
                f"使用 CloudEmbedder，API: {api_url}, "
                f"模型: {model}, 维度: {remote_dim}")
            fallback_url = _get_env("LMS_CLOUD_EMBED_FALLBACK_URL", "") or None
            return CloudEmbedder(
                api_url=api_url,
                dim=input_dim,
                remote_dim=remote_dim,
                model=model,
                timeout=30.0,     # 公网环境（embed.example.com 延迟波动 0.4-2.5s）
                retries=3,         # 公网环境（11435 间歇 reset，需重试）
                cache_size=1024,   # 手机稳定，增大缓存
                fallback_url=fallback_url,  # LAN 直连失败自动切隧道（仅本机注入）
            )
        except Exception as e:
            logger.warning(f"CloudEmbedder 失败（{e}），尝试 pretrained")
            embedder_type = 'pretrained'

    if embedder_type == 'pretrained':
        try:
            from core.sensory.embedder import PretrainedEmbedder
            model_path = _get_env(
                "LMS_PRETRAINED_MODEL", DEFAULT_PRETRAINED_MODEL_PATH)
            logger.info(f"使用 PretrainedEmbedder，模型路径: {model_path}")
            return PretrainedEmbedder(dim=input_dim, model_name=model_path)
        except ImportError as e:
            logger.warning(
                f"PretrainedEmbedder 不可用（{e}），降级为 SimpleEmbedder")
            embedder_type = 'simple'
        except Exception as e:
            logger.warning(
                f"加载预训练模型失败（{e}），降级为 SimpleEmbedder")
            embedder_type = 'simple'

    # simple（或 pretrained 降级）
    from core.sensory.embedder import SimpleEmbedder
    logger.info(f"使用 SimpleEmbedder，维度: {input_dim}")
    return SimpleEmbedder(dim=input_dim)


def build_llm_api_config(default_base_url: str, default_model: str) -> dict | None:
    """从环境变量构建 LLM API 配置。返回 None 表示未配置 API key。

    被 api/config.py 的 get_api_config() 与 mcp_memory_server.py 的
    _build_config() 共享，消除两处重复的 LLM 配置构建逻辑。

    读取环境变量:
        - DEEPSEEK_API_KEY / LMS_LLM_API_KEY: API 密钥（前者优先回退后者）
        - LMS_LLM_BASE_URL: API 基础 URL（默认 default_base_url）
        - LMS_LLM_MODEL: 模型名（默认 default_model）

    参数:
        default_base_url: 未配置环境变量时的默认 base_url。
        default_model: 未配置环境变量时的默认 model。

    返回:
        LLM API 配置字典（含 system_prompt=DEFAULT_SYSTEM_PROMPT）；
        若未提供任何 API key 则返回 None。
    """
    # 优先级：DEEPSEEK_API_KEY -> LMS_LLM_API_KEY
    api_key = _get_env("DEEPSEEK_API_KEY") or _get_env("LMS_LLM_API_KEY")
    if not api_key:
        return None
    return {
        'base_url': _get_env("LMS_LLM_BASE_URL", default_base_url),
        'api_key': api_key,
        'model': _get_env("LMS_LLM_MODEL", default_model),
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30,
        'max_retries': 3,
        'system_prompt': DEFAULT_SYSTEM_PROMPT,
    }


def get_api_config() -> dict:
    """构建并返回 API 服务用的记忆系统配置字典。

    合并 runtime.default_config()，并按环境变量覆盖以下项:
        - num_nodes / input_dim
        - embedder（实例注入）
        - activation_threshold（按 embedder 类型适配）
        - llm_api（若提供 API key 则启用，否则置空禁用 LLM）
        - auto_snapshot / snapshot_dir

    返回:
        配置字典，可直接传给 LivingMemoryLoop。
    """
    config = default_config()

    # --- 网络结构 ---
    num_nodes = int(_get_env("LMS_NUM_NODES", str(config['num_nodes'])))
    input_dim = int(_get_env("LMS_INPUT_DIM", str(config['input_dim'])))
    config['num_nodes'] = num_nodes
    config['input_dim'] = input_dim

    # --- Embedder ---
    embedder_type = _get_env("LMS_EMBEDDER", "pretrained")
    embedder = build_embedder(embedder_type, input_dim)
    config['embedder'] = embedder

    # 预训练/云端 embedder 输出幅度较小，阈值适配
    if embedder_type in ('pretrained', 'cloud') and hasattr(embedder, 'embed_text'):
        config['activation_threshold'] = 0.02

    # --- LLM API 配置 ---
    # 优先级：DEEPSEEK_API_KEY -> LMS_LLM_API_KEY
    # 共享构建逻辑见 build_llm_api_config()
    config['llm_api'] = build_llm_api_config(
        DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL)
    if config['llm_api']:
        logger.info(
            f"LLM 已配置: base_url={config['llm_api']['base_url']}, "
            f"model={config['llm_api']['model']}")
    else:
        # 未配置 API key：禁用 LLM，仅返回记忆 context
        logger.warning(
            "未检测到 DEEPSEEK_API_KEY / LMS_LLM_API_KEY，"
            "LLM 功能已禁用（/chat 仅返回记忆 context）")

    # --- 自指回路（默认关闭；dandan 看护人式决策：仅测试期开启，每日复核）---
    config['self_ref_enabled'] = (
        _get_env("LMS_SELF_REF_ENABLED", "false").lower() == "true")

    # --- 自指回路 LLM 增强蒸馏（Phase 3.3；默认关闭，开启后每 interval 轮调一次 LLM）---
    config['self_ref_llm_distill_enabled'] = (
        _get_env("LMS_SELF_REF_LLM_DISTILL_ENABLED", "false").lower() == "true")
    config['self_ref_llm_distill_interval'] = int(
        _get_env("LMS_SELF_REF_LLM_DISTILL_INTERVAL", "5"))

    # --- T2.8 算法治理开关（全部默认关闭 = 保持原行为；详见 .env.example）---
    # 惊讶度归一化：F /= ‖J‖_F（LMS_NORM_SURPRISE=1 启用）
    config['norm_surprise'] = (
        _get_env("LMS_NORM_SURPRISE", "0") == "1")
    # 回放权重钳制：replay_weight × min(surprise, cap)（0 = 不钳制）
    config['replay_surprise_cap'] = float(
        _get_env("LMS_REPLAY_SURPRISE_CAP", "0") or 0)
    # long_term_latent 归一化：consolidate 后除以 L2 范数（=1 启用）
    config['norm_latent'] = (
        _get_env("LMS_NORM_LATENT", "0") == "1")
    # meta 惰性规则：跳过 meta.update（省算力，=1 启用）
    config['meta_lazy'] = (
        _get_env("LMS_META_LAZY", "0") == "1")
    # 自述 LLM 蒸馏跳过 bus 会话（兜底防套话，=1 启用）
    config['self_ref_no_bus'] = (
        _get_env("LMS_SELF_REF_NO_BUS", "0") == "1")
    # embed 熔断器：连续 3 次失败 → 熔断 5 分钟（=1 启用）
    config['embed_circuit'] = (
        _get_env("LMS_EMBED_CIRCUIT", "0") == "1")

    # --- 论文机制 A（2026-08-17，dandan 拍板）：allostatic J 滑动设定点 ---
    # 默认关（LMS_J_ALLOSTATIC=0）→ 固定 J 行为完全不变（回滚干净）；
    # 开启后 J_target_norm 由 surprise 序列统计在线重估（Mehra 1970 innovation
    # 法 + Sterling 2012 allostasis），LMS_J_TARGET_NORM 仅作初始设定点。
    config['j_target_norm'] = float(
        _get_env("LMS_J_TARGET_NORM", "40.0") or 40.0)
    config['j_allostatic'] = (
        _get_env("LMS_J_ALLOSTATIC", "0") == "1")
    config['j_allostatic_window'] = int(
        _get_env("LMS_J_ALLOSTATIC_WINDOW", "200"))
    config['j_allostatic_k'] = float(
        _get_env("LMS_J_ALLOSTATIC_K", "2.0"))
    config['j_allostatic_step'] = float(
        _get_env("LMS_J_ALLOSTATIC_STEP", "0.5"))
    config['j_allostatic_persist'] = int(
        _get_env("LMS_J_ALLOSTATIC_PERSIST", "5"))
    config['j_allostatic_min'] = float(
        _get_env("LMS_J_ALLOSTATIC_MIN", "3.0"))
    config['j_allostatic_max'] = float(
        _get_env("LMS_J_ALLOSTATIC_MAX", "40.0"))
    config['j_allostatic_min_samples'] = int(
        _get_env("LMS_J_ALLOSTATIC_MIN_SAMPLES", "30"))
    config['j_allostatic_sat_frac'] = float(
        _get_env("LMS_J_ALLOSTATIC_SAT_FRAC", "0.9"))
    config['j_allostatic_col_act'] = int(
        _get_env("LMS_J_ALLOSTATIC_COL_ACT", "5"))

    # --- 快照配置 ---
    config['auto_snapshot'] = True
    config['auto_snapshot_interval'] = 50
    config['snapshot_dir'] = str(get_snapshot_dir())

    return config


def get_server_config() -> dict:
    """返回服务监听配置（host/port）。"""
    return {
        'host': _get_env("LMS_API_HOST", DEFAULT_API_HOST),
        'port': int(_get_env("LMS_API_PORT", str(DEFAULT_API_PORT))),
    }
