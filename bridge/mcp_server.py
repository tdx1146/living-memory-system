#!/usr/bin/env python3
"""
LMS MCP Bridge — 通过 stdin/stdout JSON-RPC 协议提供服务。

运行方式（必须用 .venv 的 python 激活 PyTorch）：
    cd /vol2/1000/AI专用/living-memory-system-cloud
    .venv/bin/python3 bridge/mcp_server.py

协议示例：
    → {"id": 1, "method": "ping"}
    ← {"id": 1, "result": "pong"}

    → {"id": 2, "method": "init", "params": {"snapshot_path": "..."}}
    ← {"id": 2, "result": {"status": "ok", "turn_count": 0}}

    → {"id": 3, "method": "process_turn", "params": {"text": "...", "llm_output": "..."}}
    ← {"id": 3, "result": {"context": "...", "entropy": 0.45, "surprise": 0.12}}

    → {"id": 4, "method": "get_status"}
    ← {"id": 4, "result": {"turn_count": 5, ...}}

    → {"id": 5, "method": "save_snapshot", "params": {"path": "..."}}
    ← {"id": 5, "result": {"status": "ok"}}

    → {"id": 6, "method": "shutdown"}
    ← {"id": 6, "result": "bye"}
"""

import sys
import json
import logging
import traceback
import os

logging.basicConfig(
    level=logging.INFO,
    format="[LMS-MCP] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lms_mcp")


class LMSMCPServer:
    """LMS MCP 服务器，通过 stdin/stdout JSON-RPC 通信。"""

    def __init__(self):
        self.loop = None

    def handle_request(self, request: dict) -> dict:
        """处理单个JSON-RPC请求。"""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "ping":
                return {"id": req_id, "result": "pong"}

            elif method == "init":
                result = self._cmd_init(params)
                result["id"] = req_id
                return result

            elif method == "process_turn":
                result = self._cmd_process_turn(params)
                result["id"] = req_id
                return result

            elif method == "get_status":
                result = self._cmd_get_status()
                result["id"] = req_id
                return result

            elif method == "save_snapshot":
                result = self._cmd_save_snapshot(params)
                result["id"] = req_id
                return result

            elif method == "shutdown":
                result = {"id": req_id, "result": "bye"}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                sys.exit(0)

            else:
                return {"id": req_id, "error": f"unknown method: {method}"}

        except Exception as e:
            logger.error(f"Error handling {method}: {e}\n{traceback.format_exc()}")
            return {"id": req_id, "error": str(e)}

    def _cmd_init(self, params: dict) -> dict:
        """初始化LivingMemoryLoop（首次调用时创建）。"""
        if self.loop is not None:
            status = self.loop.get_status()
            return {
                "result": {
                    "status": "already_initialized",
                    "turn_count": status.get("turn_count", 0),
                },
            }

        # 导入必要的模块
        lms_root = "/vol2/1000/AI专用/living-memory-system-cloud"
        sys.path.insert(0, lms_root)

        from runtime.loop import LivingMemoryLoop
        from runtime.config import RuntimeConfig
        from core.sensory.cloud_embedder import CloudEmbedder

        # 用 RuntimeConfig 构建配置
        cfg = RuntimeConfig()
        cfg.num_nodes = params.get("num_nodes", 1100)
        cfg.input_dim = params.get("input_dim", 1024)

        # 覆盖自定义嵌入器
        api_root = params.get("embed_url", "http://192.168.0.103:11435/v1/embeddings")
        embedder = CloudEmbedder(
            api_url=api_root,
            dim=cfg.input_dim,
            model=None,
            cache_size=512,
        )

        config = {
            'num_nodes': cfg.num_nodes,
            'input_dim': cfg.input_dim,
            'embedder': embedder,
            'learning_rate': cfg.learning_rate,
            'num_infer_steps': cfg.num_infer_steps,
            'temperature': cfg.temperature,
            'short_term_decay': cfg.short_term_decay,
            'long_term_decay': cfg.long_term_decay,
            'transfer_rate': cfg.transfer_rate,
            'replay_count': cfg.replay_count,
            'replay_weight': cfg.replay_weight,
            'consolidation_decay': cfg.consolidation_decay,
            'buffer_capacity': cfg.buffer_capacity,
            'precision_min': cfg.precision_min,
            'precision_max': cfg.precision_max,
            'precision_lr': cfg.precision_lr,
            'coherence_threshold': cfg.coherence_threshold,
            'min_history_length': cfg.min_history_length,
            'meta_window': cfg.meta_window,
            'max_history': cfg.max_history,
            'habituation_rate': cfg.habituation_rate,
            'activation_threshold': cfg.activation_threshold,
            'complexity_weight': cfg.complexity_weight,
            'orth_weight': cfg.orth_weight,
            'seed': cfg.seed,
            'decoder_mode': cfg.decoder_mode,
            'consolidation_interval': cfg.consolidation_interval,
        }

        # 初始化循环
        self.loop = LivingMemoryLoop(config)

        # 可选：从快照恢复
        snapshot_path = params.get("snapshot_path")
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                self.loop.load_state(snapshot_path)
                logger.info(f"从快照恢复: {snapshot_path}")
            except Exception as e:
                logger.warning(f"快照恢复失败，使用初始状态: {e}")
        elif snapshot_path:
            logger.info(f"快照不存在，从零开始: {snapshot_path}")

        status = self.loop.get_status()
        logger.info(
            f"LMS初始化完成: nodes={config['num_nodes']}, dim={config['input_dim']}, "
            f"turn_count={status['turn_count']}"
        )
        return {"result": {"status": "ok", "turn_count": status["turn_count"]}}

    def _cmd_process_turn(self, params: dict) -> dict:
        """处理一轮对话，返回记忆context。"""
        if self.loop is None:
            return {"error": "not initialized, call init first"}

        text = params.get("text", "")
        llm_output = params.get("llm_output", "")

        context = self.loop.process_turn(text, llm_output)
        status = self.loop.get_status()

        return {
            "result": {
                "context": context,
                "entropy": status.get("last_entropy", 0),
                "surprise": status.get("last_surprise", 0),
                "turn_count": status.get("turn_count", 0),
                "precision_mean": status.get("precision_mean", 0),
                "coherence": status.get("purpose_coherence", 0),
            },
        }

    def _cmd_get_status(self) -> dict:
        """获取当前状态摘要。"""
        if self.loop is None:
            return {"result": {"status": "not_initialized"}}

        status = self.loop.get_status()
        return {"result": status}

    def _cmd_save_snapshot(self, params: dict) -> dict:
        """保存当前状态到快照。"""
        if self.loop is None:
            return {"error": "not initialized"}

        path = params.get(
            "path",
            "/vol2/1000/AI专用/living-memory-system-cloud/snapshots/lms_latest.pt",
        )
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.loop.save_state(path)
        return {"result": {"status": "ok", "path": path}}

    def run(self):
        """主循环：从stdin读取JSON-RPC请求，处理，输出到stdout。"""
        logger.info("LMS MCP Server started (stdin/stdout JSON-RPC)")
        # 输出就绪信号
        sys.stdout.write(json.dumps({"event": "ready"}) + "\n")
        sys.stdout.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON: {line[:100]}")
                continue

            response = self.handle_request(request)
            if response:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main():
    server = LMSMCPServer()
    server.run()


if __name__ == "__main__":
    main()
