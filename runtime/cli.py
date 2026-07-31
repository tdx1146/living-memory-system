"""命令行入口

支持命令：run（启动记忆循环）、save（手动快照）、load（恢复）、status（查看状态）
交互式模式：读取用户输入，调用loop.process_turn

用法:
    python -m runtime.cli run [--config CONFIG] [--snapshot SNAPSHOT] [--save-on-exit PATH]
    python -m runtime.cli save PATH [--load LOAD]
    python -m runtime.cli load PATH
    python -m runtime.cli status [--snapshot SNAPSHOT]
"""

import argparse
import sys
import logging
from typing import List, Optional

from runtime.loop import LivingMemoryLoop
from runtime.config import default_config, load_config
from persistence.recovery import Recovery


def setup_logging(verbose: bool = False) -> None:
    """配置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_run(args: argparse.Namespace) -> None:
    """执行run命令：启动交互式记忆循环。"""
    # 加载配置
    config = default_config()
    if args.config:
        config.update(load_config(args.config))

    # 初始化循环
    loop = LivingMemoryLoop(config)

    # 恢复快照
    if args.snapshot:
        try:
            loop.load_state(args.snapshot)
            print(f"已从 {args.snapshot} 恢复状态")
        except RuntimeError as e:
            print(f"恢复失败: {e}")
            return

    print("活体记忆系统已启动。输入 'quit' 退出，'status' 查看状态。")
    print("-" * 60)

    while True:
        try:
            user_input = input("用户> ").strip()

            if user_input.lower() in ('quit', 'exit', 'q'):
                break

            if not user_input:
                continue

            if user_input.lower() == 'status':
                status = loop.get_status()
                for k, v in status.items():
                    print(f"  {k}: {v}")
                continue

            # 处理一轮记忆
            context = loop.process_turn(user_input)
            print(f"[记忆] {context}")

            # 如果配置了LLM，查询
            if loop.bridge is not None:
                try:
                    response = loop.query_llm(user_input)
                    print(f"助手> {response}")
                except Exception as e:
                    print(f"[LLM错误] {e}")
            else:
                print("[提示] 未配置LLM，仅展示记忆context")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\n中断")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"[错误] {e}")

    # 退出时自动保存
    if args.save_on_exit:
        try:
            loop.save_state(args.save_on_exit)
            print(f"状态已保存到 {args.save_on_exit}")
        except Exception as e:
            print(f"保存失败: {e}")


def cmd_save(args: argparse.Namespace) -> None:
    """执行save命令：手动快照。"""
    config = default_config()
    loop = LivingMemoryLoop(config)

    if args.load:
        try:
            loop.load_state(args.load)
        except RuntimeError as e:
            print(f"加载失败: {e}")
            return

    try:
        loop.save_state(args.path)
        print(f"快照已保存到 {args.path}")
    except Exception as e:
        print(f"保存失败: {e}")


def cmd_load(args: argparse.Namespace) -> None:
    """执行load命令：验证并恢复快照。"""
    recovery = Recovery()

    if recovery.validate(args.path):
        print(f"快照验证通过: {args.path}")

        # 显示快照元数据
        metadata = recovery.snapshot.get_metadata(args.path)
        print(f"  版本: {metadata.get('version', 'unknown')}")
        import datetime
        ts = metadata.get('timestamp', 0)
        if ts > 0:
            dt = datetime.datetime.fromtimestamp(ts)
            print(f"  时间: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"快照验证失败: {args.path}")


def cmd_status(args: argparse.Namespace) -> None:
    """执行status命令：查看系统状态。"""
    config = default_config()
    loop = LivingMemoryLoop(config)

    if args.snapshot:
        try:
            loop.load_state(args.snapshot)
        except RuntimeError as e:
            print(f"加载失败: {e}")
            return

    status = loop.get_status()
    print("活体记忆系统状态:")
    print("-" * 40)
    for k, v in status.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI主入口。"""
    parser = argparse.ArgumentParser(
        description='活体记忆系统 (Living Memory System) - 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m runtime.cli run
  python -m runtime.cli run --snapshot backup.pt --save-on-exit new.pt
  python -m runtime.cli save snapshot.pt
  python -m runtime.cli load snapshot.pt
  python -m runtime.cli status --snapshot snapshot.pt
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='详细日志输出')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='启动交互式记忆循环')
    run_parser.add_argument('--config', type=str, help='配置文件路径(JSON)')
    run_parser.add_argument('--snapshot', type=str, help='启动时恢复的快照路径')
    run_parser.add_argument('--save-on-exit', type=str,
                            help='退出时自动保存的路径')

    # save 命令
    save_parser = subparsers.add_parser('save', help='手动保存快照')
    save_parser.add_argument('path', type=str, help='快照保存路径')
    save_parser.add_argument('--load', type=str,
                            help='保存前先从指定快照加载')

    # load 命令
    load_parser = subparsers.add_parser('load', help='验证并恢复快照')
    load_parser.add_argument('path', type=str, help='快照文件路径')

    # status 命令
    status_parser = subparsers.add_parser('status', help='查看系统状态')
    status_parser.add_argument('--snapshot', type=str,
                               help='从快照加载后查看状态')

    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'save':
        cmd_save(args)
    elif args.command == 'load':
        cmd_load(args)
    elif args.command == 'status':
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
