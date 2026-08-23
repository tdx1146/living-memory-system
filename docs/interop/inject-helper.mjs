// inject-helper.mjs — Universal version
// No more hardcoded ports, tokens, or paths.
// Connects to OpenClaw Gateway via WebSocket, sends chat.send, then exits.
// Auto-discovers config from OpenClaw home directory.
//
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  🔒 轻如烟安全铁律（写入代码，永不遗忘）                               ║
// ║                                                                        ║
// ║  截断安全线（主人 2026-05-19 下命）：                                  ║
// ║  · 最多只允许截断当前最近的一轮对话（最近一次 user→assistant 来回）    ║
// ║  · 禁止一次性截断多条对话、禁止回溯截断                                ║
// ║  · 如需越权截断，必须主人明确授权（approved=True）                     ║
// ║                                                                        ║
// ║  Inject安全锁（自 2026-05-18）：                                       ║
// ║  · 每用户轮最多 1 次注入                                               ║
// ║  · 锁由用户下一条消息触发清除                                          ║
// ║  · 防止自我递归导致的 token 耗尽和污染                                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝

import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import os from 'node:os';

const [,, sessionKey, message, methodArg] = process.argv;
const method = methodArg || 'send'; // "send", "abort", or "history"
if (!sessionKey) {
  process.stderr.write('Usage: inject-helper.mjs <sessionKey> <message> [send|abort|history]\n');
  process.exit(1);
}
if (method !== 'abort' && method !== 'history' && !message) {
  process.stderr.write('Error: message is required for send mode\n');
  process.exit(1);
}

// ── Auto-discover OpenClaw config ──────────────────────────────────────────
function findOpenClawHome() {
  if (process.env.OPENCLAW_HOME) return process.env.OPENCLAW_HOME;
  const home = os.homedir();
  // Try standard locations
  const candidates = [
    path.join(home, '.openclaw'),
    path.join(home, '.config', 'openclaw'),
    path.join(process.env.XDG_DATA_HOME || path.join(home, '.local', 'share'), 'openclaw'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return path.join(home, '.openclaw'); // fallback default
}

function readConfig(openclawHome) {
  const jsonPath = path.join(openclawHome, 'openclaw.json');
  const yamlPath = path.join(openclawHome, 'config.yaml');
  const ymlPath = path.join(openclawHome, 'config.yml');
  
  if (fs.existsSync(jsonPath)) {
    return { format: 'json', data: JSON.parse(fs.readFileSync(jsonPath, 'utf-8')) };
  }
  // Note: yaml support would require a yaml parser; for now json only
  if (fs.existsSync(yamlPath)) {
    throw new Error('config.yaml not supported yet — convert to openclaw.json or set env vars');
  }
  if (fs.existsSync(ymlPath)) {
    throw new Error('config.yml not supported yet — convert to openclaw.json or set env vars');
  }
  return null;
}

function getConfig(obj, pathStr) {
  const parts = pathStr.split('.');
  let cur = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') return undefined;
    cur = cur[p];
  }
  return cur;
}

// ── Resolve all config ─────────────────────────────────────────────────────
const openclawHome = findOpenClawHome();
const config = readConfig(openclawHome);

// Port: env var > config > fallback
const GATEWAY_PORT = parseInt(
  process.env.GATEWAY_PORT ||
  (config && getConfig(config.data, 'gateway.port')) ||
    10
);

// Token: env var > config > fallback
const GATEWAY_TOKEN =
  process.env.GATEWAY_TOKEN ||
  (config && getConfig(config.data, 'gateway.auth.token')) ||
  '';

if (!GATEWAY_TOKEN) {
  process.stderr.write('Warning: GATEWAY_TOKEN not found. Set env var or configure gateway.auth.token.\n');
}

// Skip slow device auth (crypto.sign takes 5s on this NAS)
const dangerouslyDisableDeviceAuth = true;

// Origin: send only when dangerouslyDisableDeviceAuth is true (Gateway needs it for CSRF protection)
// When device auth is enabled, omitting Origin enables silent pairing.
const shouldSendOrigin = dangerouslyDisableDeviceAuth;

// Identity path: env var > config-based > default
let idPath = process.env.OPENCLAW_IDENTITY_PATH;
if (!idPath) {
  idPath = path.join(openclawHome, 'identity', 'device.json');
}

// ── Device identity (only if device auth is enabled) ────────────────────────
let pubB64 = null;
let deviceId = null;
let privateKey = null;

if (!dangerouslyDisableDeviceAuth) {
  if (!fs.existsSync(idPath)) {
    process.stderr.write(`Device identity not found at: ${idPath}\n`);
    process.stderr.write('If device auth is disabled, set dangerouslyDisableDeviceAuth=true in config.\n');
    process.exit(1);
  }
  const id = JSON.parse(fs.readFileSync(idPath, 'utf-8'));
  deviceId = id.deviceId;
  privateKey = crypto.createPrivateKey(id.privateKeyPem);
  const pubKey = crypto.createPublicKey(id.publicKeyPem);
  const der = pubKey.export({ type: 'spki', format: 'der' });
  const SPKI_PREFIX = Buffer.from([0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00]);
  const raw = (der.length === 44 && der.subarray(0, 12).equals(SPKI_PREFIX)) ? der.subarray(12) : der;
  pubB64 = raw.toString('base64url');
}

// ── Connection setup ───────────────────────────────────────────────────────
const instanceId = 'inject-' + crypto.randomUUID().slice(0, 8);
const wsKey = crypto.randomBytes(16).toString('base64');
const idempotencyKey = crypto.randomUUID();

const socket = net.connect(GATEWAY_PORT, '127.0.0.1');
socket.setNoDelay();

// Build HTTP upgrade headers
const upgradeHeaders = [
  'GET / HTTP/1.1',
  'Host: 127.0.0.1:' + GATEWAY_PORT,
  'Upgrade: websocket',
  'Connection: Upgrade',
  'Sec-WebSocket-Key: ' + wsKey,
  'Sec-WebSocket-Version: 13',
];
if (shouldSendOrigin) {
  upgradeHeaders.push('Origin: http://127.0.0.1:' + GATEWAY_PORT);
}
upgradeHeaders.push('', '');
socket.write(Buffer.from(upgradeHeaders.join('\r\n')));

// ── WebSocket frame helpers ────────────────────────────────────────────────
let buf = Buffer.alloc(0);
let httpUpgraded = false;

function sendFrame(data) {
  const p = Buffer.from(data, 'utf-8');
  const mk = crypto.randomBytes(4);
  const m = Buffer.alloc(p.length);
  for (let i = 0; i < p.length; i++) m[i] = p[i] ^ mk[i % 4];
  let hdr;
  if (p.length < 126) {
    hdr = Buffer.alloc(2);
    hdr[1] = 0x80 | p.length;
  } else {
    hdr = Buffer.alloc(4);
    hdr[1] = 0x80 | 126;
    hdr.writeUInt16BE(p.length, 2);
  }
  hdr[0] = 0x81;
  socket.write(Buffer.concat([hdr, mk, m]));
}

async function recvFrame(timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('timeout')); }, timeoutMs);
    const handler = (data) => {
      buf = Buffer.concat([buf, data]);
      if (!httpUpgraded) {
        const idx = buf.indexOf('\r\n\r\n');
        if (idx === -1) return;
        httpUpgraded = true;
        buf = buf.subarray(idx + 4);
      }
      while (buf.length >= 2) {
        const op = buf[0] & 0x0f;
        const masked = (buf[1] & 0x80) !== 0;
        let len = buf[1] & 0x7f;
        let h = 2;
        if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); h = 4; }
        else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); h = 10; }
        const ms = masked ? 4 : 0;
        const total = h + ms + len;
        if (buf.length < total) return;
        let p = buf.subarray(h + ms, total);
        if (masked) {
          const mk = buf.subarray(h, h + 4);
          const u = Buffer.alloc(len);
          for (let i = 0; i < len; i++) u[i] = p[i] ^ mk[i % 4];
          p = u;
        }
        buf = buf.subarray(total);
        cleanup();
        if (op === 0x8) {
          resolve({ type: 'close', status: len >= 2 ? p.readUInt16BE(0) : 0, reason: p.subarray(2).toString() });
        } else if (op === 0x1) {
          resolve({ type: 'text', data: p.toString() });
        }
      }
    };
    const cleanup = () => { clearTimeout(timer); socket.removeListener('data', handler); };
    socket.on('data', handler);
    socket.on('close', () => { cleanup(); reject(new Error('closed')); });
    socket.on('error', (e) => { cleanup(); reject(e); });
  });
}

// ── Auth flow ──────────────────────────────────────────────────────────────
async function main() {
  const t0 = Date.now();
  const chal = await recvFrame();
  console.error(`[timing] recv challenge: ${Date.now()-t0}ms`);
  if (chal.type !== 'text') throw new Error(`Expected text challenge, got ${chal.type}`);
  const nonce = JSON.parse(chal.data).payload.nonce;
  const signedAtMs = Date.now();

  // Build device signature (or skip if device auth is disabled)
  let sig = null;
  if (dangerouslyDisableDeviceAuth) {
    // Device auth is disabled — no signature needed, Gateway trusts the token
    // Use a placeholder deviceId
    deviceId = 'openclaw-control-ui';
  } else {
    sig = crypto.sign(null, Buffer.from(
      ['v3', deviceId, 'openclaw-control-ui', 'webchat',
       'operator', 'operator.admin,operator.approvals,operator.pairing',
       String(signedAtMs), GATEWAY_TOKEN, nonce, 'node.js', ''].join('|')
    ), privateKey);
  }

  const connectPayload = {
    type: 'req', id: 'ic', method: 'connect',
    params: {
      auth: { token: GATEWAY_TOKEN },
      minProtocol: 3, maxProtocol: 4,
      client: {
        id: 'openclaw-control-ui',
        displayName: '轻如烟编辑器',
        version: '1.0', platform: 'node.js',
        mode: 'webchat', instanceId,
      },
      role: 'operator',
      scopes: ['operator.admin', 'operator.approvals', 'operator.pairing'],
      device: {
        id: deviceId,
        publicKey: pubB64 || 'disabled',
        signature: sig ? sig.toString('base64url') : 'disabled',
        signedAt: signedAtMs,
        nonce,
      },
      caps: ['tool-events'],
      userAgent: 'inject-helper/1.0',
      locale: 'zh-CN',
    },
  };

  sendFrame(JSON.stringify(connectPayload));
  console.error(`[timing] connect sent: ${Date.now()-t0}ms`);

  const conn = await recvFrame();
  console.error(`[timing] recv connect ack: ${Date.now()-t0}ms`);
  if (conn.type === 'close') {
    throw new Error(`Closed: ${conn.status} ${conn.reason}`);
  }
  const connRes = JSON.parse(conn.data);
  if (!connRes.ok) throw new Error('Connect failed: ' + JSON.stringify(connRes.error));

  // Send the RPC message (chat.send, chat.abort, or chat.history)
  if (method === 'abort') {
    sendFrame(JSON.stringify({
      type: 'req', id: 'ia-' + crypto.randomUUID().slice(0, 8), method: 'chat.abort',
      params: { sessionKey },
    }));
    // Wait briefly for the abort acknowledgment
    try {
      const ack = await recvFrame(5000);
      if (ack.type === 'text') {
        const r = JSON.parse(ack.data);
        if (!r.ok) throw new Error('Abort failed: ' + JSON.stringify(r.error));
      }
    } catch (e) { /* timeout OK */ }
    process.stdout.write(JSON.stringify({ ok: true }));
    
  } else if (method === 'history') {
    sendFrame(JSON.stringify({
      type: 'req', id: 'ih-' + crypto.randomUUID().slice(0, 8), method: 'chat.history',
      params: { sessionKey, limit: 500 },
    }));
    try {
      // Collect all response frames (chat.history streams events + RPC response)
      let messages = null;
      const deadline = Date.now() + 10000;
      while (Date.now() < deadline) {
        const res = await recvFrame(2000);
        if (res.type === 'text') {
          const d = JSON.parse(res.data);
          // RPC response
          if (d.type === 'res' || d.id) {
            if (d.ok && d.messages) messages = d.messages;
            break;
          }
          // Event with payload containing messages
          if (d.payload && d.payload.messages) messages = d.payload.messages;
          if (d.payload && d.payload.message && !messages) messages = [d.payload.message];
          // Direct messages field
          if (d.messages) messages = d.messages;
          // If this is a terminal event, we might have enough
          if (d.event === 'chat.history' || d.type === 'event') continue;
        } else if (res.type === 'close') {
          break;
        }
      }
      if (messages) {
        process.stdout.write(JSON.stringify({ ok: true, messages }));
      } else {
        throw new Error('No messages received');
      }
    } catch (e) {
      throw new Error('History request failed: ' + e.message);
    }
    
  } else {
    sendFrame(JSON.stringify({
      type: 'req', id: 'is-' + crypto.randomUUID().slice(0, 8), method: 'chat.send',
      params: { sessionKey, message, deliver: true, idempotencyKey },
    }));

    // ── Reliability fix (2026-08-03): wait for gateway ack instead of fire-and-forget ──
    // Root cause: gateway event loop can block up to ~2.9s under load; a message frame
    // may be sent but never processed while the UI already shows success → message loss.
    // Fix: wait for the RPC `res` frame (gateway confirms receipt/handling). Timeout 3s
    // (>2.9s peak block). On timeout report failure + exit non-zero so edit-web.py can retry.
    const SEND_ACK_TIMEOUT_MS = 3000;
    const ackDeadline = Date.now() + SEND_ACK_TIMEOUT_MS;
    let acked = false;
    while (Date.now() < ackDeadline) {
      let res;
      try {
        res = await recvFrame(Math.max(1, ackDeadline - Date.now()));
      } catch (e) {
        break; // timeout (or socket closed) — treat as not-acked
      }
      if (res.type === 'text') {
        let d;
        try { d = JSON.parse(res.data); } catch { continue; }
        // RPC response frame (type:'res' or has our id) — gateway acked receipt/handling
        if (d && d.type === 'res' || (d && d.id && String(d.id).startsWith('is-'))) {
          if (d.ok) {
            acked = true;
          } else {
            throw new Error('gateway 未确认: ' + JSON.stringify(d.error || 'rpc error'));
          }
          break;
        }
        // Otherwise it's an event/stream frame — keep waiting for the RPC response
      } else if (res.type === 'close') {
        throw new Error('gateway 未确认: connection closed (status ' + res.status + ')');
      }
    }
    if (!acked) {
      throw new Error('gateway 未确认（3s 超时）');
    }
    process.stdout.write(JSON.stringify({ ok: true }));
  }

  socket.end();
}

main().catch(e => {
  process.stderr.write('Error: ' + e.message + '\n');
  try { socket.end(); } catch {}
  process.stdout.write(JSON.stringify({ ok: false, error: e.message }));
  process.exit(1);
});
