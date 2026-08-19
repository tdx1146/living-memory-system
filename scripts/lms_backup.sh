#!/usr/bin/env bash
# =============================================================================
# LMS 活体记忆系统 - 备份轮换脚本（阶段 2 / T2.4，总体方案 §3.8）
# -----------------------------------------------------------------------------
# 职责：
#   1) 15 分钟级：snapshots/ 增量镜像复制 → backups/snapshots-15min/
#      （rsync -a --delete：幂等、保留 mtime、增量传输；镜像即"最新快照集"）
#   2) 每小时：snapshots/ 归档 tar.zst → backups/hourly/（保留 BACKUP_KEEP_HOURS 小时）
#   3) 每日 02:30：tar.zst 全量备份（snapshots/ + data/ + logs/，排除临时文件）
#      → backups/daily/lms-YYYYMMDD.tar.zst，保留 BACKUP_KEEP_DAYS 天滚动
#   4) 备份元数据：backups/MANIFEST.jsonl（时间/类型/大小/hash）追加式记录
#   5) 跨机复制（可选）：REMOTE_BACKUP 配置 rsync 目标，默认空 = 仅本机
#
# 工程级特性：
#   * 幂等：任意时刻可安全重跑（inc 是 rsync 镜像；daily 覆盖当日归档）
#   * 失败告警：任何阶段失败 → 非零退出码 + 追加 logs/lms_alerts.jsonl（CRIT）
#   * 并发保护：flock 非阻塞锁；上一轮未结束则本轮跳过（exit 0，不误报）
#   * 不依赖第三方：bash + rsync + tar(zstd) + sha256sum，全部系统自带
#
# 用法：
#   lms_backup.sh inc|incremental|--quick    # 15 分钟级增量镜像（cron */15）
#   lms_backup.sh hourly|--hourly            # 每小时归档 tar.zst（cron 0 * * * *）
#   lms_backup.sh daily|full|--daily         # 每日全量 tar.zst（cron 02:30）
#   lms_backup.sh status                     # 查看备份状态（MANIFEST 尾部 + 目录大小）
#
# 环境变量（均可覆盖，默认值见下）：
#   LMS_BACKUP_DIR      备份根目录（默认 = LMS 上级/backups/lms，自动推导）
#   REMOTE_BACKUP       rsync 跨机目标（如 user@host:/path，默认空=本机）
#   BACKUP_KEEP_DAYS    每日归档保留天数（默认 30，总体方案 §3.8）
#   BACKUP_KEEP_HOURS   每小时归档保留小时数（默认 168 = 7 天，总体方案 §3.8）
#   RSYNC_BWLIMIT       跨机 rsync 带宽限制 KB/s（默认 0=不限）
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMS_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载项目 .env（若存在）：备份路径/远端目标等配置可放 .env。
# 用 set -a 导出为环境变量（与 MCP/API 启动范式一致）。
if [ -f "$LMS_HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$LMS_HOME/.env"
    set +a
fi

# 备份根目录：自动推导（与总体方案 §3.8 一致）
# 备份根目录：优先环境变量；未设置时从脚本位置推导（开源友好，无硬编码路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMS_ROOT_DEFAULT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="${LMS_BACKUP_DIR:-$(dirname "$LMS_ROOT_DEFAULT")/backups/lms}"
INC_DIR="$BACKUP_ROOT/snapshots-15min"
HOURLY_DIR="$BACKUP_ROOT/hourly"
DAILY_DIR="$BACKUP_ROOT/daily"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-30}"
KEEP_HOURS="${BACKUP_KEEP_HOURS:-168}"   # 7 天（总体方案 §3.8）
REMOTE_BACKUP="${REMOTE_BACKUP:-}"
RSYNC_BWLIMIT="${RSYNC_BWLIMIT:-0}"   # KB/s，0 = 不限

SNAPSHOT_DIR="${LMS_SNAPSHOT_DIR:-$LMS_HOME/snapshots}"
LOG_FILE="$LMS_HOME/logs/lms_backup.log"
ALERT_FILE="$LMS_HOME/logs/lms_alerts.jsonl"
MANIFEST="$BACKUP_ROOT/MANIFEST.jsonl"
LOCK_FILE="$LMS_HOME/run/lms_backup.lock"
LAST_SYNC_MARK="$BACKUP_ROOT/.last-sync-inc"

HOST="$(hostname -s 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
log()  { echo "[lms_backup][$(date '+%F %T')] $*" >> "$LOG_FILE"; echo "[lms_backup] $*"; }
warn() { echo "[lms_backup][WARN][$(date '+%F %T')] $*" >> "$LOG_FILE"; echo "[lms_backup][WARN] $*" >&2; }
fail() { echo "[lms_backup][ERROR][$(date '+%F %T')] $*" >> "$LOG_FILE"; echo "[lms_backup][ERROR] $*" >&2; }

# 告警落盘（尽力而为：告警文件写失败不阻断备份流程本身）
alert() {
    local msg="$1"
    if [ -d "$LMS_HOME/logs" ]; then
        printf '{"ts":"%s","level":"CRIT","source":"lms_backup","host":"%s","msg":"%s"}\n' \
            "$(date -Iseconds)" "$HOST" "$msg" >> "$ALERT_FILE" 2>/dev/null || true
    fi
}

# MANIFEST 追加一行（追加式 JSONL；值均为本脚本自产，无用户输入注入面）
manifest() {
    local type="$1" target="$2" size="$3" hash="$4"
    mkdir -p "$BACKUP_ROOT"
    printf '{"ts":"%s","type":"%s","target":"%s","size":%s,"sha256":%s,"host":"%s"}\n' \
        "$(date -Iseconds)" "$type" "$target" \
        "${size:-null}" "${hash:-null}" "$HOST" >> "$MANIFEST"
}

# 并发保护：flock 非阻塞；拿不到锁 = 上一轮仍在跑 → 跳过（exit 0，幂等语义）
acquire_lock() {
    mkdir -p "$LMS_HOME/run"
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "上一轮备份仍在运行，本轮跳过（flock 非阻塞）"
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# 15 分钟级增量镜像
# ---------------------------------------------------------------------------
run_incremental() {
    local t0 rc files size
    t0="$(date +%s)"
    log "== 增量镜像开始（snapshots/ → $INC_DIR） =="

    if [ ! -d "$SNAPSHOT_DIR" ]; then
        warn "快照目录不存在（$SNAPSHOT_DIR），跳过增量镜像（fail-open）"
        return 0
    fi
    mkdir -p "$INC_DIR"

    # rsync 镜像：-a 保留属性/mtime（增量传输），--delete 与源严格一致，
    # 排除 *.lock（fcntl 锁占位文件，零字节无价值，避免每次抖动）。
    # --timeout 防 NFS/网络卡死；失败时保留上一轮镜像（不删目标）。
    if ! rsync -a --delete --timeout=300 --exclude='*.lock' \
            "$SNAPSHOT_DIR/" "$INC_DIR/" ; then
        fail "增量镜像 rsync 失败（退出码 $?）"
        return 1
    fi

    # 同步完成标记（镜像目录之外，防 --delete 误删；供监控判新鲜度）
    date -Iseconds > "$LAST_SYNC_MARK" 2>/dev/null || true

    # 元数据：镜像文件数 + 总字节（hash 对本层无意义，记 null 保持诚实）
    files="$(find "$INC_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')"
    size="$(du -sb "$INC_DIR" 2>/dev/null | awk '{print $1}')"
    manifest "inc15" "snapshots-15min" "${size:-0}" "null"

    local dur=$(( $(date +%s) - t0 ))
    log "== 增量镜像完成：${files} 个文件，${size} 字节，耗时 ${dur}s =="
    return 0
}

# ---------------------------------------------------------------------------
# 每小时归档 tar.zst（T2.4 补：--hourly，保留 BACKUP_KEEP_HOURS 小时）
# ---------------------------------------------------------------------------
run_hourly() {
    local t0 stamp archive tmp_archive rc
    t0="$(date +%s)"
    stamp="$(date +%Y%m%d_%H%M)"
    archive="$HOURLY_DIR/lms-hourly-$stamp.tar.zst"
    tmp_archive="$HOURLY_DIR/.lms-hourly-$stamp.tar.zst.tmp.$$"
    log "== 每小时归档开始 → $archive =="

    if [ ! -d "$SNAPSHOT_DIR" ]; then
        warn "快照目录不存在（$SNAPSHOT_DIR），跳过每小时归档（fail-open）"
        return 0
    fi
    mkdir -p "$HOURLY_DIR"

    # 仅打包 snapshots/（轻量高频层）；data/ 与 logs/ 由每日全量覆盖。
    # 先写临时文件再原子 mv：中途失败不会留下半个归档冒充本轮备份。
    if ! ( cd "$LMS_HOME" && tar --zstd -cf "$tmp_archive" \
            --exclude='*.lock' --exclude='*.tmp' --exclude='*.swp' \
            --warning=no-file-changed --warning=no-file-ignored \
            snapshots ); then
        rm -f "$tmp_archive" 2>/dev/null || true
        fail "每小时归档 tar.zst 打包失败"
        return 1
    fi
    mv -f "$tmp_archive" "$archive"

    # 完整性冒烟：zstd 帧校验 + tar 列表可读
    if ! zstd -t "$archive" >/dev/null 2>&1; then
        fail "完整性校验失败（zstd -t）: $archive"
        return 1
    fi
    if ! tar --zstd -tf "$archive" >/dev/null 2>&1; then
        fail "完整性校验失败（tar -tf）: $archive"
        return 1
    fi

    # 元数据：大小 + sha256
    local size hash
    size="$(stat -c %s "$archive" 2>/dev/null || echo 0)"
    hash="$(sha256sum "$archive" | awk '{print $1}')"
    manifest "hourly" "hourly/$(basename "$archive")" "$size" "\"$hash\""

    # 滚动保留：只删 lms-hourly-*.tar.zst，超过 KEEP_HOURS 小时的删除
    find "$HOURLY_DIR" -maxdepth 1 -name 'lms-hourly-*.tar.zst' \
        -mmin "+$((KEEP_HOURS * 60))" -delete 2>/dev/null || true
    local kept
    kept="$(find "$HOURLY_DIR" -maxdepth 1 -name 'lms-hourly-*.tar.zst' 2>/dev/null | wc -l | tr -d ' ')"
    log "== 每小时归档完成：$(basename "$archive")（${size} 字节，sha256=${hash:0:16}…），保留 ${kept} 份 =="
    return 0
}

# ---------------------------------------------------------------------------
# 每日全量 tar.zst
# ---------------------------------------------------------------------------
run_daily() {
    local t0 stamp archive tmp_archive rc
    t0="$(date +%s)"
    stamp="$(date +%Y%m%d)"
    archive="$DAILY_DIR/lms-$stamp.tar.zst"
    tmp_archive="$DAILY_DIR/.lms-$stamp.tar.zst.tmp.$$"
    log "== 每日全量开始 → $archive =="

    if [ ! -d "$SNAPSHOT_DIR" ]; then
        warn "快照目录不存在（$SNAPSHOT_DIR），跳过全量备份（fail-open）"
        return 0
    fi
    mkdir -p "$DAILY_DIR"

    # 全量打包：snapshots/ + data/ + logs/，排除临时/锁/缓存文件。
    # 相对路径（在 LMS_HOME 内执行 tar）→ 归档内路径干净，恢复直接解包。
    # 先写临时文件再原子 mv：中途失败不会留下半个归档冒充当日备份。
    if ! ( cd "$LMS_HOME" && tar --zstd -cf "$tmp_archive" \
            --exclude='*.lock' --exclude='*.tmp' --exclude='*.swp' \
            --exclude='__pycache__' --exclude='*.pyc' \
            snapshots data logs ); then
        rm -f "$tmp_archive" 2>/dev/null || true
        fail "tar.zst 打包失败"
        return 1
    fi
    mv -f "$tmp_archive" "$archive"

    # 完整性冒烟：zstd 帧校验 + tar 列表可读
    if ! zstd -t "$archive" >/dev/null 2>&1; then
        fail "完整性校验失败（zstd -t）: $archive"
        return 1
    fi
    if ! tar --zstd -tf "$archive" >/dev/null 2>&1; then
        fail "完整性校验失败（tar -tf）: $archive"
        return 1
    fi

    # 元数据：大小 + sha256
    local size hash
    size="$(stat -c %s "$archive" 2>/dev/null || echo 0)"
    hash="$(sha256sum "$archive" | awk '{print $1}')"
    manifest "daily" "daily/$(basename "$archive")" "$size" "\"$hash\""

    # 滚动保留：删除 KEEP_DAYS 天前的每日归档（只删 lms-*.tar.zst）
    find "$DAILY_DIR" -maxdepth 1 -name 'lms-*.tar.zst' -mtime "+$KEEP_DAYS" \
        -delete 2>/dev/null || true
    local kept
    kept="$(find "$DAILY_DIR" -maxdepth 1 -name 'lms-*.tar.zst' 2>/dev/null | wc -l | tr -d ' ')"
    log "== 每日全量完成：$(basename "$archive")（${size} 字节，sha256=${hash:0:16}…），保留 ${kept} 份 =="
    return 0
}

# ---------------------------------------------------------------------------
# 跨机复制（可选）：REMOTE_BACKUP 非空时执行
# ---------------------------------------------------------------------------
run_remote() {
    [ -n "$REMOTE_BACKUP" ] || return 0
    log "== 跨机复制开始 → $REMOTE_BACKUP =="
    # -az 归档+压缩；--delete 与本地备份根一致；BatchMode 免交互（密钥登录）
    # --bwlimit 限带宽防拖垮生产链路；失败即告警（不吞错）
    if ! rsync -az --delete --timeout=600 --bwlimit="$RSYNC_BWLIMIT" \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
            "$BACKUP_ROOT/" "$REMOTE_BACKUP/"; then
        fail "跨机复制失败（REMOTE_BACKUP=$REMOTE_BACKUP）"
        return 1
    fi
    manifest "remote" "REMOTE_BACKUP" "null" "null"
    log "== 跨机复制完成 =="
    return 0
}

# ---------------------------------------------------------------------------
# 状态查看
# ---------------------------------------------------------------------------
run_status() {
    echo "== LMS 备份状态 =="
    echo "备份根目录 : $BACKUP_ROOT"
    echo "快照目录   : $SNAPSHOT_DIR"
    echo "远端目标   : ${REMOTE_BACKUP:-（未配置，仅本机）}"
    echo "每日保留   : ${KEEP_DAYS} 天"
    echo
    if [ -f "$MANIFEST" ]; then
        echo "-- MANIFEST 最近 8 条 --"
        tail -n 8 "$MANIFEST"
    else
        echo "（MANIFEST 尚不存在，尚未执行过备份）"
    fi
    echo
    [ -d "$INC_DIR" ] && echo "15min 镜像 : $(du -sh "$INC_DIR" 2>/dev/null | awk '{print $1}')（$(find "$INC_DIR" -type f | wc -l | tr -d ' ') 文件）" || echo "15min 镜像 : 未执行"
    [ -d "$HOURLY_DIR" ] && echo "每小时归档 : $(du -sh "$HOURLY_DIR" 2>/dev/null | awk '{print $1}')（$(find "$HOURLY_DIR" -name 'lms-hourly-*.tar.zst' | wc -l | tr -d ' ') 份）" || echo "每小时归档 : 未执行"
    [ -d "$DAILY_DIR" ] && echo "每日归档   : $(du -sh "$DAILY_DIR" 2>/dev/null | awk '{print $1}')（$(find "$DAILY_DIR" -name 'lms-*.tar.zst' | wc -l | tr -d ' ') 份）" || echo "每日归档   : 未执行"
    return 0
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
main() {
    local action="${1:-inc}"
    mkdir -p "$BACKUP_ROOT"
    acquire_lock

    case "$action" in
        inc|incremental|--quick)
            run_incremental
            rc=$?
            if [ $rc -eq 0 ]; then
                run_remote || rc=1
            fi
            ;;
        hourly|--hourly)
            run_hourly
            rc=$?
            if [ $rc -eq 0 ]; then
                run_remote || rc=1
            fi
            ;;
        daily|full|--daily)
            run_daily
            rc=$?
            if [ $rc -eq 0 ]; then
                run_remote || rc=1
            fi
            ;;
        status)
            run_status
            return $?
            ;;
        *)
            echo "用法: $0 {inc|incremental|--quick|hourly|--hourly|daily|full|--daily|status}" >&2
            return 2
            ;;
    esac

    if [ $rc -ne 0 ]; then
        alert "备份失败（action=$action, rc=$rc），请检查 $LOG_FILE"
        fail "备份流程失败（rc=$rc）—— 告警已写入 $ALERT_FILE"
    fi
    return $rc
}

main "$@"
