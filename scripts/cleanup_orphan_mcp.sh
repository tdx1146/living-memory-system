#!/usr/bin/env bash
# =============================================================================
# cleanup_orphan_mcp.sh — MCP 孤儿进程回收（阶段1-B / T1.8，P1-3 配套）
# =============================================================================
# 背景：历史上有 MCP 子进程（mcp_memory_server / lms_http_mcp / shouji_memory_mcp）
# 的父进程（gateway/IDE）退出后，子进程被 reparent 到 PID 1，成为孤儿：
#   - 旧直连版 mcp_memory_server 每个 ~425MB，孤儿累积吃内存（曾见 3 个 root/旧实例）；
#   - 阶段1-B 薄桥化后单进程 ~30MB，但孤儿仍需回收，防累积。
#
# 判定规则（安全优先，宁漏杀不误杀）：
#   1. cmdline 匹配目标 MCP 脚本（绝对路径或脚本名）；
#   2. 孤儿判定（默认）：PPID == 1（父进程已死被 init 收养）；
#      --any-parent 模式：父进程 PID 在 /proc 中不存在也算孤儿；
#   3. 存活时长 >= MIN_AGE 秒（默认 600s，避免误杀刚启动、PPID 暂显 1 的进程）；
#   4. 排除：本脚本自身、--exclude 指定的 PID、当前用户的 shell 链。
#
# 用法：
#   scripts/cleanup_orphan_mcp.sh                 # 只打印待清理清单（默认 dry-run）
#   scripts/cleanup_orphan_mcp.sh --yes           # 打印并实际 kill（SIGTERM，先弱后强）
#   scripts/cleanup_orphan_mcp.sh --yes --any-parent
#   scripts/cleanup_orphan_mcp.sh --min-age 60 --exclude 12345,67890
#
# 退出码：0=正常执行（有/无候选均可）；1=参数错误。
# =============================================================================
set -u

# ---- 配置 ----------------------------------------------------------------
# 匹配的 MCP 脚本（按 cmdline 子串匹配，覆盖旧直连版与薄桥版）
PATTERNS=("mcp_memory_server.py" "lms_http_mcp.py" "shouji_memory_mcp")
# 默认最小存活时长（秒）：低于该时长不视为可回收孤儿（防误杀刚拉起的新实例）
MIN_AGE=600
YES=0
ANY_PARENT=0
EXCLUDE=""

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    echo "选项:"
    echo "  --yes            实际 kill（默认仅打印清单，dry-run）"
    echo "  --any-parent     父进程 PID 不存在即视为孤儿（默认仅 PPID==1）"
    echo "  --min-age SEC    最小存活秒数（默认 600）"
    echo "  --exclude PIDS   额外排除 PID 列表（逗号分隔）"
    exit 1
}

# ---- 参数解析 --------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --yes) YES=1 ;;
        --any-parent) ANY_PARENT=1 ;;
        --min-age) MIN_AGE="${2:?--min-age 需要数值}"; shift ;;
        --exclude) EXCLUDE="${2:?--exclude 需要 PID 列表}"; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1" >&2; usage ;;
    esac
    shift
done

# 组装排除集（本脚本自身 + 调用方 shell 链 + --exclude）
EXCLUDE_SET="$$"
[ -n "$EXCLUDE" ] && EXCLUDE_SET="$EXCLUDE_SET,$EXCLUDE"
# 把父进程链（bash 调用链）也排除，避免误杀脚本自己的宿主 shell
_p=$PPID
while [ "$_p" != "1" ] && [ -n "$_p" ]; do
    EXCLUDE_SET="$EXCLUDE_SET,$_p"
    _p=$(awk '{print $4}' "/proc/$_p/stat" 2>/dev/null || echo 1)
done

_now=$(date +%s)
CANDIDATES=0
KILLED=0

echo "[cleanup-orphan-mcp] $(date '+%Y-%m-%d %H:%M:%S') 扫描开始 (dry-run=$([ $YES -eq 0 ] && echo yes || echo no), min_age=${MIN_AGE}s, any_parent=$ANY_PARENT)"

for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    [ -r "/proc/$pid/cmdline" ] || continue
    # 自身与排除集
    case ",$EXCLUDE_SET," in
        *",$pid,"*) continue ;;
    esac

    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed 's/ *$//')
    [ -n "$cmdline" ] || continue

    # 1) cmdline 匹配任一 MCP 模式
    matched=0
    for pat in "${PATTERNS[@]}"; do
        case "$cmdline" in
            *"$pat"*) matched=1; break ;;
        esac
    done
    [ $matched -eq 1 ] || continue

    # 2) 孤儿判定：PPID==1（默认）；--any-parent：父进程不存在也算
    ppid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null || echo 0)
    orphan=0
    if [ "$ppid" = "1" ]; then
        orphan=1
    elif [ "$ANY_PARENT" -eq 1 ] && [ ! -d "/proc/$ppid" ]; then
        orphan=1
    fi
    [ $orphan -eq 1 ] || continue

    # 3) 存活时长门槛（starttime 是自系统启动的时钟 tick，需加 /proc/stat 的 btime）
    start_ticks=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || echo 0)
    hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
    [ "$hz" -gt 0 ] 2>/dev/null || hz=100
    boot_epoch=$(awk '/^btime /{print $2}' /proc/stat 2>/dev/null || echo 0)
    start_epoch=$(( boot_epoch + start_ticks / hz ))
    age=$(( _now - start_epoch ))
    [ "$age" -ge "$MIN_AGE" ] || continue

    # 4) 读取进程详情（rss 由 stat rss 字段换算，字段 24）
    rss_kb=$(( $(awk '{print $24}' "/proc/$pid/stat" 2>/dev/null || echo 0) * 4 ))
    user=$(stat -c '%U' "/proc/$pid" 2>/dev/null || echo "?")
    start=$(date -d "@$start_epoch" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "?")

    CANDIDATES=$((CANDIDATES + 1))
    printf "  [待清理] pid=%s ppid=%s user=%s rss≈%sMB age=%ss start=%s\n" \
        "$pid" "$ppid" "$user" "$(( rss_kb / 1024 ))" "$age" "$start"
    printf "           cmd: %s\n" "$cmdline"

    if [ "$YES" -eq 1 ]; then
        # 先 SIGTERM（优雅），1.5s 后仍存活再 SIGKILL
        kill -TERM "$pid" 2>/dev/null || { echo "  [跳过] pid=$pid 已不存在"; continue; }
        sleep 1.5
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
            echo "  [已强杀] pid=$pid (SIGTERM 后仍存活 → SIGKILL)"
        else
            echo "  [已终止] pid=$pid (SIGTERM 优雅退出)"
        fi
        KILLED=$((KILLED + 1))
    fi
done

echo "[cleanup-orphan-mcp] 扫描结束: 候选 $CANDIDATES 个, 已 kill $KILLED 个"
exit 0
