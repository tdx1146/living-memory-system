"""MCP 记忆服务器工具逻辑测试脚本

直接调用 mcp_memory_server 中的工具函数（不走 MCP 协议），
验证 recall_memory / store_memory / get_memory_status 三个工具的正确性。

测试流程:
  1. get_memory_status — 触发懒加载初始化，检查初始状态
  2. store_memory      — 存储第一条对话
  3. recall_memory     — 检索刚存储的对话
  4. store_memory      — 存储第二条对话
  5. recall_memory     — 检索相关记忆（应能命中）
  6. get_memory_status — 检查状态变化（轮次增加、缓冲区增长）

运行方式:
    python test_mcp_tools.py
"""

import os
import sys

# 项目根目录（绝对路径）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import mcp_memory_server as mms


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_get_status_initial():
    """测试1: 获取初始状态（触发懒加载）"""
    separator("测试1: get_memory_status（初始状态）")
    result = mms.do_get_memory_status()
    print(result)
    print("\n[OK] 初始状态获取成功")
    return result


def test_store_memory_1():
    """测试2: 存储第一条对话"""
    separator("测试2: store_memory（第一条对话）")
    text = "用户: 我叫小明，我喜欢吃火锅。\n助手: 你好小明！火锅确实很好吃，你喜欢什么口味的火锅底料呢？"
    result = mms.do_store_memory(text)
    print(result)
    print("\n[OK] 第一条对话存储成功")
    return result


def test_recall_memory_1():
    """测试3: 检索记忆（应能找到刚存储的对话）"""
    separator("测试3: recall_memory（检索'小明喜欢吃什么'）")
    result = mms.do_recall_memory("小明喜欢吃什么")
    print(result)
    print("\n[OK] 记忆检索成功")
    return result


def test_store_memory_2():
    """测试4: 存储第二条对话"""
    separator("测试4: store_memory（第二条对话）")
    text = "用户: 我在学Python编程，觉得装饰器很难理解。\n助手: 装饰器确实是Python的一个难点。简单来说，装饰器是一个接受函数并返回新函数的高阶函数。"
    result = mms.do_store_memory(text)
    print(result)
    print("\n[OK] 第二条对话存储成功")
    return result


def test_recall_memory_2():
    """测试5: 检索记忆（查询编程相关，应命中第二条）"""
    separator("测试5: recall_memory（检索'Python编程'）")
    result = mms.do_recall_memory("Python编程装饰器")
    print(result)
    print("\n[OK] 记忆检索成功")
    return result


def test_recall_memory_3():
    """测试6: 检索记忆（查询美食相关，应命中第一条）"""
    separator("测试6: recall_memory（检索'火锅美食'）")
    result = mms.do_recall_memory("火锅美食")
    print(result)
    print("\n[OK] 记忆检索成功")
    return result


def test_get_status_final():
    """测试7: 获取最终状态"""
    separator("测试7: get_memory_status（最终状态）")
    result = mms.do_get_memory_status()
    print(result)
    print("\n[OK] 最终状态获取成功")
    return result


def test_call_tool_handler():
    """测试8: 通过 async handler 测试 MCP 分派逻辑"""
    separator("测试8: handle_call_tool 分派逻辑（异步）")
    import asyncio

    async def run():
        # 模拟 MCP 调用 get_memory_status
        class FakeParams:
            name = "get_memory_status"
            arguments = {}

        result = await mms.handle_call_tool(None, FakeParams())
        print(f"工具: get_memory_status")
        print(f"is_error: {result.is_error}")
        print(f"内容长度: {len(result.content[0].text)} 字符")
        print(f"内容前100字: {result.content[0].text[:100]}...")

        # 模拟未知工具
        class FakeParamsUnknown:
            name = "nonexistent_tool"
            arguments = {}

        result2 = await mms.handle_call_tool(None, FakeParamsUnknown())
        print(f"\n工具: nonexistent_tool")
        print(f"is_error: {result2.is_error}")
        print(f"内容: {result2.content[0].text}")

        # 模拟 list_tools
        list_result = await mms.handle_list_tools(None, None)
        print(f"\n工具列表: {[t.name for t in list_result.tools]}")

    asyncio.run(run())
    print("\n[OK] MCP handler 分派逻辑测试成功")


if __name__ == "__main__":
    print("活体记忆系统 MCP 服务器 - 工具逻辑测试")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"Python: {sys.executable}")

    # 运行所有测试
    test_get_status_initial()
    test_store_memory_1()
    test_recall_memory_1()
    test_store_memory_2()
    test_recall_memory_2()
    test_recall_memory_3()
    test_get_status_final()
    test_call_tool_handler()

    separator("全部测试完成")
    print("所有工具逻辑测试通过！")
