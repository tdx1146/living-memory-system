# OpenClaw Gateway WebSocket 协议笔记

> 2026-05-18 逆向整理，by 轻如烟

## 概述

OpenClaw Gateway (v2026.3.13) 使用 WebSocket 作为 RPC 通信协议，端口 **22881**（默认 18789，此实例配置为 22881）。
认证采用 **challenge-response + Ed25519 设备签名** 机制。

## 认证流程

### 1. HTTP 升级

```http
GET / HTTP/1.1
Host: 127.0.0.1:22881
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: <random base64>
Sec-WebSocket-Version: 13
Origin: http://127.0.0.1:22881  (非必须，但传了有助于本地识别)
```

### 2. Gateway 推送 challenge (紧接在 101 之后)

```json
{
  "type": "event",
  "event": "connect.challenge",
  "payload": {
    "nonce": "uuid-string",
    "ts": 1779120742950
  }
}
```

### 3. 客户端回复 connect RPC

请求格式：

```json
{
  "type": "req",
  "id": "unique-id",
  "method": "connect",
  "params": {
    "auth": { "token": "... 或 password" },
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "openclaw-control-ui",   // 必须使用允许的 client id
      "displayName": "任意名称",
      "version": "1.0",
      "platform": "node.js",         // 会被签名
      "mode": "webchat"
    },
    "role": "operator",
    "scopes": ["operator.admin", "operator.approvals", "operator.pairing"],
    "device": {
      "id": "sha256(ed25519-raw-pubkey).hexdigest()",
      "publicKey": "base64url(ed25519-raw-32-bytes)",
      "signature": "base64url(ed25519.sign(payload))",
      "signedAt": 1779120742950,
      "nonce": "同 challenge 的 nonce"
    },
    "caps": ["tool-events"],
    "userAgent": "...",
    "locale": "zh-CN"
  }
}
```

### 4. Gateway 回复

成功：
```json
{
  "type": "res",
  "id": "相同 request id",
  "ok": true,
  "payload": {
    "type": "hello-ok",
    "protocol": 3,
    "server": { "version": "2026.3.13", "connId": "..." },
    "features": { "methods": ["..."] }
  }
}
```

失败（最常见的几种）：
```
// client.id 不合法
{ "ok": false, "error": { "code": "INVALID_REQUEST", "message": "at /client/id: must be ..." } }

// 设备未配对（第一次连接）
{ "ok": false, "error": { "code": "NOT_PAIRED", "message": "pairing required", "details": { "code": "PAIRING_REQUIRED", ... } } }

// 签名无效
{ "ok": false, "error": { "code": "INVALID_REQUEST", "message": "device signature invalid" } }
```

## 设备签名 (v3 payload)

### 签名内容

```
v3|deviceId|clientId|clientMode|role|scopes|signedAtMs|token|nonce|platform|deviceFamily
```

以 `|` 连接（pipe-separated），所有字段必填（空字段用空字符串 `""`）。

参数来源：
- `deviceId`: sha256(raw_ed25519_pubkey).hexdigest()
- `clientId`: connectParams.client.id
- `clientMode`: connectParams.client.mode
- `role`: "operator"
- `scopes`: scopes.join(",") → "operator.admin,operator.approvals,operator.pairing"
- `signedAtMs`: Date.now()
- `token`: auth.token (或 password, deviceToken)
- `nonce`: 来自 challenge
- `platform`: connectParams.client.platform
- `deviceFamily`: connectParams.client.deviceFamily (可能为 "")

### 公钥处理

Ed25519 SPKI DER 格式前缀：`0x302a300506032b6570032100`（12 字节）
原始 32 字节在 SPKI 之后：`der.subarray(12)`

`device.id = sha256(raw_32_bytes).hexdigest()`
`device.publicKey = base64url(raw_32_bytes)`

### 签名算法

```javascript
// JS crypto
const signature = crypto.sign(null, Buffer.from(payload, 'utf-8'), privateKey);
const sigB64url = signature.toString('base64url');
```

```python
# Python cryptography
from cryptography.hazmat.primitives.asymmetric import ed25519
sig = private_key.sign(payload.encode())
sig_b64url = base64.urlsafe_b64encode(sig).rstrip(b'=').decode()
```

## 允许的 client IDs（GatewayClientIdSchema）

```
- webchat-ui
- openclaw-control-ui  ← 推荐用于 inject 工具
- webchat
- cli                  ← 需要先配对
- gateway-client
- openclaw-macos
- openclaw-ios
- openclaw-android
- node-host
- test
- fingerprint
- openclaw-probe
```

允许的 client modes：
```
webchat, cli, ui, backend, node, probe, test
```

## 本地自动配对 (Silent Pairing)

必须同时满足以下条件：
1. `client.id === "openclaw-control-ui"` 或 `isWebchatClient`
2. 连接来自 localhost（isLocalClient）
3. 无浏览器 Origin header（或已是控制 UI）
4. 原因是 "not-paired" 或 "scope-upgrade"

**推荐做法**：使用 `openclaw-control-ui` + `webchat` 模式 + `operator.pairing` scope。

## 发送消息 (chat.send)

```json
{
  "type": "req",
  "id": "unique-id",
  "method": "chat.send",
  "params": {
    "sessionKey": "main",
    "message": "用户消息文本",
    "deliver": false,
    "idempotencyKey": "random-uuid"
  }
}
```

- `deliver: false` 表示不通过消息通道发送（只是注入到会话中让 AI 处理）
- AI 处理是异步的，response 可能很久才到；建议不等待直接断开

## 历史弯路

1. ❌ 尝试直接 Python 实现 `websocket-client` + 原始 WS 协议 → 缺少 `type: "req"` 字段
2. ❌ 使用 `cli` + `cli` 模式 → 签名通过但设备未配对，返回 PAIRING_REQUIRED
3. ❌ 使用 `v2` payload 格式 → OpenClaw v2026.3.13 强制 v3
4. ❌ recv 时不处理 HTTP 101 响应 → 被当作 WS close frame 解析
5. ✅ 使用 `openclaw-control-ui` + `webchat` + `operator.pairing` → silent pairing 自动完成

## RPC 方法探测结果（2026-05-26 09:00）

| 方法 | 状态 | 说明 |
|------|------|------|
| chat.history | ✅ 可用 | 返回 sessionKey/main 的消息列表，含 messages[] 数组 |
| session.list | ✅ 可用 | 同 chat.history，别名 |
| rpc.discover | ❌ 不存在 | 无 introspection |
| session.info | ❌ 不存在 | |
| transcript.get | ❌ 不存在 | |

### chat.history 调用格式

```json
{
  "type": "req",
  "method": "chat.history",
  "params": { "sessionKey": "main", "limit": 500 }
}
```

返回：
```json
{
  "type": "res",
  "ok": true,
  "payload": {
    "sessionKey": "main",
    "sessionId": "51118b30-...",
    "messages": [
      { "role": "user", "content": [...] },
      { "role": "assistant", "content": [...] },
      ...
    ]
  }
}
```

### 连接参数（dangerouslyDisableDeviceAuth: true）
```
Origin: http://127.0.0.1:22881
auth.token: <GATEWAY_TOKEN>
device.id: "openclaw-control-ui"
device.publicKey: "disabled"
device.signature: "disabled"
```

### 方案1 可行性
✅ 编辑器可以通过 WebSocket RPC 调用 `chat.history` 获取会话数据，
不再直接读 JSONL 文件，彻底消除并发冲突。
需要在前端或编辑器中层做 WebSocket 连接池 + 缓存策略。
