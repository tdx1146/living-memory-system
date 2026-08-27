#!/usr/bin/env bash
# =============================================================================
# Agent OS PID 文件探活修复（T1.12 / P1-8）
# -----------------------------------------------------------------------------
# 背景：Agent OS start_all.sh 用「kill -0 $PID」判断"已在运行"，但 run/*.pid
#   指向 DEAD PID 时判断失效 → 重跑起重复实例、端口占用冲突（P1-8 实证）。
#   Agent OS 是独立目录独立仓库 → 不改其源码；本脚本作为外部运维辅助，
#   从 Agent OS 之外修正其 PID 文件，并桥接 systemd 接管场景。
#
# 功能：
#   1) 扫描 $AGENT_OS_HOME/run/*.pid 与 iso-sand/data/*.pid；
#   2) 校验每个 PID：必须存活 **且** cmdline 匹配预期模式
#      （防 PID 复用：死进程的 PID 被新进程复用 → kill -0 通过但 cmdline 不符）；
#   3) 已死/不匹配 → 删除该 PID 文件（打印原因）；
#   4) --adopt：对关键服务按端口回填真实 PID（ss 端口解析，start_all.sh 同款），
#      使 systemd 接管后 start_all.sh 的 kill -0 检查通过 → 自动跳过不重复拉起。
#
# 用法：
#   bash scripts/fix_pid_probe.sh             # 扫描并清理过期 PID 文件
#   bash scripts/fix_pid_probe.sh --adopt     # 清理 + 按端口回填真实 PID
#   bash scripts/fix_pid_probe.sh --dry-run   # 只打印将要执行的动作，不落盘
#
# 退出码：0=全部正常  1=存在已修复/需人工关注项（供脚本化调用判断）
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMS_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
# Agent OS 定位：优先环境变量，否则按同布局兄弟目录相对推导（零硬编码）
AGENT_OS_HOME="${AGENT_OS_HOME:-$(cd "$LMS_HOME/../Agent OS" 2>/dev/null && pwd)}"
if [ -z "$AGENT_OS_HOME" ]; then
    echo "[fix-pid][WARN] 无法定位 Agent OS（请设置 AGENT_OS_HOME 环境变量）" >&2
    exit 1
fi
# 统一配置：端口等从 Agent OS/env.local 读取
if [ -f "$AGENT_OS_HOME/env.local" ]; then
    set -a; . "$AGENT_OS_HOME/env.local"; set +a
fi
SANDGLASS_API_PORT="${SANDGLASS_API_PORT:-17333}"
LMS_API_PORT="${LMS_API_PORT:-8190}"
GLUE_PORT="${GLUE_PORT:-19000}"
DRY_RUN=0
ADOPT=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --adopt)   ADOPT=1 ;;
        *) echo "未知参数: $arg（支持 --dry-run / --adopt）" >&2; exit 2 ;;
    esac
done

FIXED=0
KEPT=0
ADOPTED=0
OK_COUNT=0

log()  { echo "[fix-pid] $*"; }
act()  { echo "[fix-pid][ACTION] $*"; FIXED=$((FIXED + 1)); }
warn() { echo "[fix-pid][WARN] $*" >&2; }

[ -d "$AGENT_OS_HOME" ] || { warn "AGENT_OS_HOME 不存在: $AGENT_OS_HOME"; exit 1; }

# ---------------------------------------------------------------------------
# 预期 cmdline 模式表（按 PID 文件名匹配；未列出的文件仅做存活校验）
# ---------------------------------------------------------------------------
pattern_for() {
    local name="$1"
    case "$name" in
        lms_api.pid)            echo "api.run" ;;
        glue_server.pid)        echo "glue_server" ;;
        sandglass_http_api.pid) echo "sandglass_http_api" ;;
        verify_daemon.pid)      echo "verify_daemon" ;;
        scheduler.pid)          echo "scheduler" ;;
        consumer.pid)           echo "consumer" ;;
        *)                      echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# 端口 → 真实监听 PID 映射（--adopt 用；start_all.sh 同款 ss 解析）
# ---------------------------------------------------------------------------
port_for() {
    local name="$1"
    case "$name" in
        lms_api.pid)            echo "$LMS_API_PORT" ;;
        glue_server.pid)        echo "$GLUE_PORT" ;;
        sandglass_http_api.pid) echo "$SANDGLASS_API_PORT" ;;
        *)                      echo "" ;;
    esac
}

pid_listening_on() {
    local port="$1"
    ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1
}

# ---------------------------------------------------------------------------
# 单个 PID 文件校验
#   → keep | remove | adopt(port)
# ---------------------------------------------------------------------------
check_pidfile() {
    local file="$1" name pid pattern port live_pid
    name="$(basename "$file")"
    pid="$(cat "$file" 2>/dev/null | tr -d '[:space:]')"
    pattern="$(pattern_for "$name")"

    # 无效（空/非数字）或已死：优先 --adopt 按端口回填真实 PID（systemd 接管桥接），
    # 无端口映射/端口无监听者才删除。
    if [ -z "$pid" ] || [ -n "${pid//[0-9]/}" ] || ! kill -0 "$pid" 2>/dev/null; then
        if [ "$ADOPT" -eq 1 ]; then
            port="$(port_for "$name")"
            if [ -n "$port" ]; then
                live_pid="$(pid_listening_on "$port")"
                if [ -n "$live_pid" ]; then
                    ADOPTED=$((ADOPTED + 1))
                    act "回填 $file：PID '${pid}' 无效/已死，端口 $port 实际监听 PID $live_pid"
                    [ "$DRY_RUN" -eq 0 ] && echo "$live_pid" > "$file"
                    return
                fi
            fi
        fi
        act "删除 $file：PID '${pid}' 无效/已死"
        [ "$DRY_RUN" -eq 0 ] && rm -f "$file"
        return
    fi

    # cmdline 匹配校验（防 PID 复用）
    if [ -n "$pattern" ]; then
        local cmd
        cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
        case "$cmd" in
            *"$pattern"*)
                log "OK $file：PID $pid 存活且匹配 '$pattern'"
                OK_COUNT=$((OK_COUNT + 1))
                ;;
            *)
                act "删除 $file：PID $pid 存活但 cmdline 不匹配 '$pattern'（PID 复用）"
                [ "$DRY_RUN" -eq 0 ] && rm -f "$file"
                return
                ;;
        esac
    else
        log "OK $file：PID $pid 存活（无模式约束）"
        OK_COUNT=$((OK_COUNT + 1))
    fi

    # --adopt：按端口回填真实 PID（systemd 接管桥接，PID 已正确则保持）
    if [ "$ADOPT" -eq 1 ]; then
        port="$(port_for "$name")"
        [ -z "$port" ] && return
        live_pid="$(pid_listening_on "$port")"
        if [ -n "$live_pid" ] && [ "$live_pid" != "$pid" ]; then
            ADOPTED=$((ADOPTED + 1))
            act "回填 $file：端口 $port 实际监听 PID $live_pid（原记录 $pid）"
            [ "$DRY_RUN" -eq 0 ] && echo "$live_pid" > "$file"
        elif [ -n "$live_pid" ]; then
            KEPT=$((KEPT + 1))
        else
            warn "端口 $port 无监听者，$file 保留原 PID（服务未运行？）"
        fi
    fi
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
[ "$DRY_RUN" -eq 1 ] && log "DRY-RUN 模式：只打印动作，不修改任何文件"
log "扫描 $AGENT_OS_HOME 的 PID 文件..."

FILES=()
[ -d "$AGENT_OS_HOME/run" ] && FILES+=("$AGENT_OS_HOME/run"/*.pid)
[ -d "$AGENT_OS_HOME/iso-sand/data" ] && FILES+=("$AGENT_OS_HOME/iso-sand/data"/*.pid)

TOTAL=${#FILES[@]}
for file in "${FILES[@]}"; do
    # glob 无匹配时保留字面 *.pid，跳过
    [ -f "$file" ] || continue
    check_pidfile "$file"
done

echo ""
if [ "$TOTAL" -eq 0 ]; then
    log "未发现 PID 文件（$AGENT_OS_HOME/run 与 iso-sand/data 均空或不存在）"
elif [ "$FIXED" -gt 0 ]; then
    log "检查 $TOTAL 个 PID 文件：$OK_COUNT 正常，修复 $FIXED 个（回填 $ADOPTED，保持 $KEPT）"
    exit 1   # 有修复 → 非零退出，提示调用方（start_all.sh 前跑一次）
else
    log "检查 $TOTAL 个 PID 文件：全部正常（$OK_COUNT）"
fi
exit 0
