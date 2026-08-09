#!/usr/bin/env bash
# =============================================================================
# LMS API 控制脚本（T1.9 / P0-8 配套）
# -----------------------------------------------------------------------------
# 兼容两种运行模式：
#   1) systemd 用户单元模式（首选）：lms-api.service 已安装且用户管理器可用
#      → start/stop/restart/status 全部委派 systemctl --user（不写 PID 文件）；
#   2) 非 systemd 模式（过渡期/沙盒/无 systemd 环境）：
#      setsid 独立会话 + PID 文件 + 端口/进程双探活（镜像 start_all.sh 范式）。
#
# 探活原则（P1-8 教训）：
#   * kill 旧进程前先校验 PID 存活 **且** /proc/PID/cmdline 匹配 api.run
#     （防过期 PID 文件 + PID 复用导致的误杀/误判"已在运行"）；
#   * 不依赖 PID 文件也能定位进程（/proc 扫描 + 端口探测兜底）。
#
# 用法：
#   lms_ctl.sh status                  # 查看状态（自动选择模式）
#   lms_ctl.sh start                   # 启动
#   lms_ctl.sh stop                    # 优雅停机（SIGTERM → 30s 落盘 → KILL 兜底）
#   lms_ctl.sh restart                 # 重启
#   lms_ctl.sh <cmd> --force-manual    # 强制非 systemd 模式
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMS_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
UNIT_NAME="lms-api.service"
RUN_DIR="$LMS_HOME/run"
PID_FILE="$RUN_DIR/lms_api.pid"
LOG_FILE="$LMS_HOME/logs/lms_api.log"
API_HOST="${LMS_API_HOST:-127.0.0.1}"
API_PORT="${LMS_API_PORT:-8190}"
HEALTH_URL="http://$API_HOST:$API_PORT/health"
STOP_GRACE=35   # 与 systemd TimeoutStopSec 一致（server.py 30s 落盘 + 裕量）

FORCE_MANUAL=0
for arg in "$@"; do
    [ "$arg" = "--force-manual" ] && FORCE_MANUAL=1
done
ACTION="${1:-status}"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
log()  { echo "[lms_ctl] $*"; }
warn() { echo "[lms_ctl][WARN] $*" >&2; }
fail() { echo "[lms_ctl][ERROR] $*" >&2; exit 1; }

# systemd 用户管理器可用 + 单元已安装？
_systemd_ok() {
    [ "$FORCE_MANUAL" -eq 1 ] && return 1
    command -v systemctl >/dev/null 2>&1 || return 1
    [ -n "${XDG_RUNTIME_DIR:-}" ] || export XDG_RUNTIME_DIR="/run/user/$(id -u)"
    [ -d "$XDG_RUNTIME_DIR" ] || return 1
    systemctl --user is-system-running >/dev/null 2>&1 || return 1
    # is-enabled 对未安装单元返回 1（"No such file or directory"）
    systemctl --user is-enabled "$UNIT_NAME" >/dev/null 2>&1 || return 1
    return 0
}

# /proc 扫描：cmdline 含指定模式的进程 PID 列表（最权威，不依赖 PID 文件）
_find_pids_by_pattern() {
    local pattern="$1" pid cmd
    for pid in /proc/[0-9]*; do
        pid="${pid#/proc/}"
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
        case "$cmd" in
            *"$pattern"*) echo "$pid" ;;
        esac
    done
}

# 校验 PID 文件中的 PID：存活且 cmdline 匹配 api.run（防复用误判）
_pidfile_valid() {
    local pid
    [ -f "$PID_FILE" ] || return 1
    pid="$(cat "$PID_FILE" 2>/dev/null)" || return 1
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "api\.run" || return 1
    return 0
}

# 当前 API 进程 PID 集合（PID 文件 ∪ /proc 扫描，去重）
_api_pids() {
    local pids pid
    pids="$(_find_pids_by_pattern "api.run")"
    if _pidfile_valid; then
        pid="$(cat "$PID_FILE")"
        case " $pids " in *" $pid "*) ;; *) pids="$pids $pid" ;; esac
    fi
    echo "$pids"
}

_api_running() {
    [ -n "$(_api_pids)" ] && return 0
    # 端口兜底（进程刚起、cmdline 尚未稳定时）
    (exec 3<>"/dev/tcp/$API_HOST/$API_PORT") 2>/dev/null && { exec 3>&-; return 0; }
    return 1
}

_health_ok() {
    curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1
}

_wait_healthy() {
    local i
    for i in $(seq 1 20); do
        _health_ok && return 0
        sleep 3
    done
    return 1
}

# ---------------------------------------------------------------------------
# 动作实现（非 systemd 模式）
# ---------------------------------------------------------------------------
manual_start() {
    if _api_running; then
        log "API 已在运行（PID: $(_api_pids | tr '\n' ' ')），跳过"
        return 0
    fi
    [ -f "$LMS_HOME/.env" ] || fail "缺少 $LMS_HOME/.env（含密钥与嵌入配置）"
    mkdir -p "$RUN_DIR" "$LMS_HOME/logs"
    log "启动（非 systemd 模式，日志: $LOG_FILE）..."
    # 镜像 start_all.sh 范式：子 shell 内 cd + source .env + setsid 独立会话
    ( cd "$LMS_HOME" && set -a && . "$LMS_HOME/.env" && set +a && \
      setsid "$LMS_HOME/.venv/bin/python" -m api.run --host "$API_HOST" --port "$API_PORT" \
      >> "$LOG_FILE" 2>&1 < /dev/null & echo $! > "$PID_FILE" )
    if ! _wait_healthy; then
        warn "启动后 60s 内未通过 /health，进程 PID=$(cat "$PID_FILE" 2>/dev/null)，日志尾部:"
        tail -n 20 "$LOG_FILE" 2>/dev/null
        return 1
    fi
    # setsid 可能 fork，按端口解析真实 PID 回填（start_all.sh 同款）
    local real_pid
    real_pid="$(ss -tlnp 2>/dev/null | grep ":$API_PORT " | grep -oP 'pid=\K[0-9]+' | head -1)"
    [ -n "$real_pid" ] && echo "$real_pid" > "$PID_FILE"
    log "启动成功 PID=$(cat "$PID_FILE")，/health OK"
}

manual_stop() {
    local pids pid
    pids="$(_api_pids)"
    if [ -z "$pids" ]; then
        log "未发现运行中的 API 进程（PID 文件与 /proc 均无匹配）"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE" && log "清理过期 PID 文件 $PID_FILE"
        return 0
    fi
    log "发送 SIGTERM（优雅停机，最多等 ${STOP_GRACE}s 落盘）: PID $pids"
    for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
    # 等待进程退出（server.py 30s 落盘看门狗内应完成）
    local i
    for i in $(seq 1 "$STOP_GRACE"); do
        [ -z "$(_api_pids)" ] && break
        sleep 1
    done
    pids="$(_api_pids)"
    if [ -n "$pids" ]; then
        warn "优雅停机超时，SIGKILL 兜底: PID $pids"
        for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
        sleep 1
    fi
    rm -f "$PID_FILE"
    log "已停止"
}

manual_status() {
    local pids pid_file_state="" port_state="" health_state="" line
    pids="$(_api_pids)"
    [ -f "$PID_FILE" ] && _pidfile_valid && pid_file_state="有效" || pid_file_state="过期/缺失"
    if (exec 3<>"/dev/tcp/$API_HOST/$API_PORT") 2>/dev/null; then
        exec 3>&-
        port_state="监听中"
    else
        port_state="未监听"
    fi
    _health_ok && health_state="OK" || health_state="异常/未响应"
    echo "LMS API 状态（非 systemd 模式）:"
    echo "  进程: ${pids:-无}"
    echo "  PID 文件 ($PID_FILE): $pid_file_state"
    echo "  端口 $API_HOST:$API_PORT: $port_state"
    echo "  /health: $health_state"
    if [ -n "$pids" ]; then
        for pid in $pids; do
            read -r line < <(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
            echo "  [PID $pid] $line"
        done
    fi
    [ -n "$pids" ] && return 0 || return 1
}

# ---------------------------------------------------------------------------
# 动作分发
# ---------------------------------------------------------------------------
if _systemd_ok; then
    case "$ACTION" in
        start)
            log "systemd 模式：daemon-reload + start $UNIT_NAME"
            systemctl --user daemon-reload
            systemctl --user start "$UNIT_NAME" || fail "systemctl start 失败"
            systemctl --user is-active "$UNIT_NAME" >/dev/null || fail "单元未进入 active"
            log "已启动（active）"
            ;;
        stop)
            log "systemd 模式：stop $UNIT_NAME（触发优雅停机落盘）"
            systemctl --user stop "$UNIT_NAME" || fail "systemctl stop 失败"
            log "已停止"
            ;;
        restart)
            log "systemd 模式：restart $UNIT_NAME"
            systemctl --user restart "$UNIT_NAME" || fail "systemctl restart 失败"
            log "已重启"
            ;;
        status)
            echo "LMS API 状态（systemd 模式）:"
            systemctl --user --no-pager status "$UNIT_NAME" || true
            ;;
        *)
            fail "未知动作: $ACTION（支持 start|stop|restart|status）"
            ;;
    esac
else
    case "$ACTION" in
        start)   manual_start ;;
        stop)    manual_stop ;;
        restart) manual_stop && manual_start ;;
        status)  manual_status ;;
        *)       fail "未知动作: $ACTION（支持 start|stop|restart|status）" ;;
    esac
fi
