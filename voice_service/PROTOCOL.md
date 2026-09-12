# Voice Service WebSocket 协议（v1）

面向**其他 agent / 前端接入方**的接口说明：把 Voice Service 当作用户的"耳朵和嘴" ——
你通过 WebSocket 接收语音识别结果，并把要说的文本交给它朗读；麦克风、唤醒词、
VAD、TTS、播放队列、连续对话窗口全部由服务端负责。

- 参考实现（可复制到你的项目）：[`voice_service/protocol.py`](voice_service/protocol.py)（零第三方依赖，仅标准库）
- 本仓库的 hermes 接入示例：[`hermes_gateway_plugin/`](../hermes_gateway_plugin/)
- 架构与部署（本机实测记录）：[`../docs/voice-service-websocket-architecture.md`](../docs/voice-service-websocket-architecture.md)

> 约定：本文件描述的是**线上行为**；常量名与帧结构以 `protocol.py` 为准。协议版本 `v=1`。

---

## 1. 连接与鉴权

| 项目 | 说明 |
|---|---|
| 端点 | `ws://<host>:<port>`，默认 `ws://127.0.0.1:8765` |
| 鉴权 | 服务端配置 `service.token` 非空时，握手 HTTP 头需带 `Authorization: Bearer <token>`；不匹配返回 **HTTP 401** |
| 探活 | `GET /healthz` → `200 ok`（`text/plain`）；其它非 WebSocket 请求 → `404` |
| 并发 | **v1 单客户端**：已有客户端时，第二个连接被拒（关闭码 `4090 already occupied`） |
| 心跳 | 应用层 `ping`/`pong`（WebSocket 协议层 ping 已关闭）；服务端在 `heartbeat_timeout_sec`（默认 60s）内收不到**任何**帧就断开 |
| 帧上限 | 单帧 ≤ 1 MiB |
| 编码 | UTF-8 JSON；一条 WebSocket 消息 = 一帧（不转义非 ASCII） |

**关闭码**

| 码 | 含义 |
|---|---|
| `4090` | 已有客户端占用（单客户端语义） |
| `4091` | 连接后 10s 内未收到 `hello` |
| `4092` | 首帧不是 `hello` |
| `1000` | 客户端 `bye` 正常收尾 |

---

## 2. 帧格式（envelope）

每个帧都是一行 JSON 对象：

```json
{
  "v": 1,
  "type": "speak",
  "seq": 7,
  "client_id": "my-agent",
  "ts": 1710000000.123,
  "data": { }
}
```

| 字段 | 说明 |
|---|---|
| `v` | 协议版本，固定 `1`；其它值按非法帧处理 |
| `type` | 帧类型（见下表） |
| `seq` | 命令序号（客户端自增）。服务端在 `ack` 中原样回填；服务端事件的 `seq` 通常为 `null`（`pong` 除外） |
| `client_id` | 客户端标识（注册用，v1 仅记录/日志） |
| `ts` | Unix 秒时间戳（浮点） |
| `data` | 帧负载对象；缺失或非对象时按 `{}` 处理 |

非法帧（JSON 解析失败 / `v` 不符 / 缺 `type`）不会断开连接：服务端回一条
`ack{seq:0, ok:false, error:"bad frame"}`。

---

## 3. 握手

```
客户端 ── hello ──► 服务端
客户端 ◄─ welcome ─ 服务端
```

**`hello`（客户端首帧，必须 10s 内发出）**

```json
{"v":1,"type":"hello","seq":1,"client_id":"my-agent","ts":0,
 "data":{"client_id":"my-agent","caps":{"interim":true}}}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `client_id` | 是 | 客户端标识（也可放在 envelope 上） |
| `caps` | 否 | 能力声明；**当前服务端仅记录日志**。`interim`=希望收到 ASR 中间结果（该事件尚未实现，见 §5）；`voiceprint`=支持说话人标注 |

**`welcome`（服务端应答）**

```json
{"v":1,"type":"welcome","seq":null,"client_id":"","ts":0,
 "data":{"ok":true,
         "my_wakewords":[{"name":"泡泡","keywords":["泡泡管家","你好泡泡"]},
                         {"name":"小布","keywords":["小布小布"]}],
         "version":1}}
```

`my_wakewords` 是服务端**实际**助手表（唤醒词 ↔ 助手映射由服务端 `config.yaml`
的 `wake_word.assistants` 维护）；客户端可用它了解当前有哪些助手，无需自己配置词表。

---

## 4. 客户端 → 服务端（命令）

| type | data | 说明 |
|---|---|---|
| `speak` | `{id, wake, kind, segments, turn_seq}` | **唯一播放入口**，见 §6 |
| `control` | `{action, ...}` | `close_window`：关闭连续对话窗口、回到唤醒词监听（**客户端应把"不朗读"结束标记翻译成这一帧**，例如 `[FINISH]`）；`dismiss_reply` / `reload_kws` 为预留（当前仅记日志）；未知 action 记日志后仍 `ack ok` |
| `interrupt` | `{}` | 停 TTS、清空播报队列、停等待音（物理打断通常由服务端本地处理，这一帧用于客户端主动打断） |
| `ping` | `{}` | 心跳，服务端回 `pong`（同 `seq`） |
| `bye` | `{reason?}` | 优雅断开：先 `ack`，服务端再以 `1000` 关闭 |

每条命令都会收到一条 `ack`：

```json
{"v":1,"type":"ack","seq":7,"client_id":"","ts":0,
 "data":{"seq":7,"ok":true,"error":null}}
```

> 注意：`speak` 因"迟到的旧轮"被丢弃时，`ack` 仍是 `ok:true`（静默丢弃，见 §6）。

---

## 5. 服务端 → 客户端（事件）

| type | data | 触发时机 |
|---|---|---|
| `welcome` | `{ok, my_wakewords, version}` | `hello` 应答（§3） |
| `ack` | `{seq, ok, error?}` | 每条命令的应答 |
| `wake_word` | `{keyword, wake}` | 唤醒词命中（服务端已本地播提示音/打断）。两者均为**助手名**（服务端 KWS 词表的 `@` 后缀） |
| `asr_result` | `{text, wake, message_id, turn_seq}` | 用户一段话识别完成（**主事件**）；`text` 见下方说明（声纹启用时含说话人前缀） |
| `speak_done` | `{id, kind, ok}` | 一段 `speak` 播报完成（`ok=false` 表示 TTS/播放失败） |
| `pong` | `{}` | `ping` 应答（同 `seq`） |
| `asr_interim` | `{text, wake}` | **预留，当前不发送**（需 `caps.interim`） |
| `asr_sentence` | `{text, speaker_id?, speaker_label?, start_ms?, end_ms?}` | **预留，当前不发送** |
| `error` | `{code, message}` | **预留，当前不发送**（错误以日志/关闭码体现） |

**`asr_result` 说明**

```json
{"v":1,"type":"asr_result","seq":null,"client_id":"","ts":0,
 "data":{"text":"[爸爸 (ID: 100)] 今天天气怎么样？","wake":"小布",
         "message_id":"voice-1710000000123","turn_seq":12}}
```

- `wake`：本轮属于哪个助手（你用它做会话路由，例如 `chat_id = "wake:" + wake`）；
- `turn_seq`：服务端每轮自增的回合号 —— 发 `speak` 时**原样带回**，服务端据此丢弃迟到的旧轮回复；
- `text`：识别文本；服务端启用声纹（`voiceprint.enabled`）时**按句换行**，每句带
  `[说话人名字 (ID: 编号)]` 前缀，例如：
  `"[爸爸 (ID: 100)] 打开客厅灯\n[未知 (ID: 101)] 你好"`。
  名字取 `voiceprint.speaker_names`（`spk_100` → 真实姓名）映射，未映射时显示「未知」；
  编号是声纹库 id 的数字部分（`spk_100` → `100`），同一说话人跨轮次稳定，可用于区分不同人。
  名字与编号始终成对出现；识别不出且未注册时只有「未知」（无编号）。
  **客户端把 `text` 原样作为用户消息交给 agent 即可，无需再解析或重排**；
- `message_id`：`voice-<epoch_ms>`，仅用于日志串联。

---

## 6. 播报语义（`speak.kind` / `turn_seq`）

```json
{"v":1,"type":"speak","seq":9,"client_id":"my-agent","ts":0,
 "data":{"id":"t-123","wake":"小布","kind":"final",
         "segments":[["happy","查好啦！"],["neutral","明天晴，20~30℃。"]],
         "turn_seq":12}}
```

| 字段 | 说明 |
|---|---|
| `id` | 本次播报标识，会原样出现在 `speak_done` 中 |
| `wake` | 助手名（来自 `asr_result` / `wake_word`），决定会话归属与去旧范围 |
| `kind` | `final`（最终回答）/ `interim`（中间轮，如工具调用过程中的话）/ `raw`（按最终处理，兼容值） |
| `segments` | `[[情绪, 文本], ...]`；情绪取固定集合：`neutral` / `sad` / `happy` / `angry` / `fear` / `story` / `poetry` / `sajiao` / `disgusted` / `amaze` / `exciting` / `aojiao` / `jieshuo`，留空字符串表示不指定；不在集合内的 `(xxx)` 不会被当作情绪标记（按原文朗读）；`[]` 或全空文本 → 只回 `speak_done`、不出声 |
| `turn_seq` | 建议回填 `asr_result.turn_seq`；`0`/省略表示不做旧轮判定 |

**服务端播后行为（客户端无需自己判断）**

| kind | 播完后的行为 |
|---|---|
| `final` / `raw` | 播通知音 → 进入**连续对话窗口**（`conversation_window.timeout_sec`，本仓库 `config.yaml` 为 5.0s）：用户可直接接着说，无需再喊唤醒词；窗口超时 → 播告别语并回到待机 |
| `interim` | 若服务端仍在等待该轮最终回复 → **恢复等待提示音与回复超时兜底**；不播通知音、不进对话窗口、不触发告别语 |

其它规则：

- **迟到的旧轮丢弃**：同一 `wake` 下，`speak.turn_seq` 小于已消费轮次 → 丢弃（用于打断后旧回复的兜底）；
- **被打断会话的回复**（`speak.wake` ≠ 服务端当前活跃助手，即用户在助手 A 思考/调工具期间
  唤醒了助手 B）：A 的 `interim` **直接丢弃**（不再打扰当前对话）；A 的 `final` 保留并**排队**
  （等当前播放结束后播出），且播放时会在首个非空分段前加 `我是<A>，` 前缀，说明这是谁的回答。
  因此客户端**必须让 `wake` 准确反映该回复所属的会话**（按会话归属，而不是"当前活跃助手"）；
- **等待回复超时**：`asr_result` 上行后 `wait_reply_timeout_sec`（默认 45s）内没有任何 `speak` → 服务端播告别语收尾；
- **结束对话**：客户端若判断本轮"不需要朗读"（例如模型只调用了播放控制类工具并返回 `[FINISH]`），请发 `control{action:"close_window"}`，**不要**把该标记当文本 `speak` 进来（服务端不会解析它）；
- 服务端**不校验**你是否在"对话窗口内"—— `speak` 到达即按上述规则播报。

---

## 7. 最小接入示例（Python）

```python
import asyncio, json, websockets

URL = "ws://127.0.0.1:8765"
CID = "my-agent"

def frame(type_, seq, data):
    return json.dumps({"v": 1, "type": type_, "seq": seq,
                       "client_id": CID, "ts": 0, "data": data},
                      ensure_ascii=False)

async def heartbeat(ws, period=20):
    seq = 1000
    while True:
        await asyncio.sleep(period)
        seq += 1
        await ws.send(frame("ping", seq, {}))

async def main():
    async with websockets.connect(URL) as ws:
        await ws.send(frame("hello", 1, {"client_id": CID, "caps": {"interim": True}}))
        print("welcome:", json.loads(await ws.recv())["data"])
        asyncio.create_task(heartbeat(ws))

        seq = 1
        async for raw in ws:
            f = json.loads(raw)
            t, d = f["type"], f.get("data") or {}
            if t == "asr_result" and d.get("text"):
                # TODO: 交给你的 agent 处理；回复时用下面两行下发
                seq += 1
                await ws.send(frame("speak", seq, {
                    "id": f"t-{d['turn_seq']}", "wake": d["wake"], "kind": "final",
                    "segments": [["neutral", "好的，我来处理。"]],
                    "turn_seq": d["turn_seq"]}))
            elif t == "speak_done":
                print("播报完成:", d)
            elif t == "pong":
                pass

asyncio.run(main())
```

接入要点：`hello` 必须是首帧；**任何**帧都算心跳，但请保持周期 `ping`；同一时刻只连一个客户端；
用 `wake` 作为会话键、`turn_seq` 回填 `speak`。

---

## 8. 服务端相关配置（`voice_service/config.yaml`）

| 配置 | 作用 |
|---|---|
| `service.host` / `service.port` | 监听地址与端口（默认 `127.0.0.1:8765`） |
| `service.token` | 空 = 仅本机；非空时客户端需 `Authorization: Bearer <token>` |
| `service.ping_interval_sec` | 参考值（服务端当前不主动发 `ping`；客户端心跳周期自定，见 §7 示例 20s） |
| `service.heartbeat_timeout_sec` | 服务端在此时间内收不到任何帧则断开（默认 60s） |
| `service.wait_reply_timeout_sec` | `asr_result` 后等待 `speak` 的超时，超时播告别语（默认 45s） |
| `conversation_window.timeout_sec` | `final` 播完后的连续对话窗口时长 |
| `wake_word.assistants` | 助手表（`name` = 上行事件里的 `wake`）；服务端据此生成 KWS 词表 |
| `voiceprint.enabled` | 是否在 `asr_result.text` 中按句加 `[名字 (ID: 编号)]` 前缀 |

---

## 9. 兼容性与演进

- `v=1` 内**只做加法**：新增帧类型/`data` 字段不影响既有客户端；客户端应忽略未知 `type` 与未知字段；
- `asr_interim` / `asr_sentence` / `error` 与 `control{dismiss_reply|reload_kws}` 已占位（见 `protocol.py` 常量），实现后无需变更协议版本；
- 常量与工具函数（`make_frame` / `dumps` / `loads` / `ack_for` / `parse_emotion_segments`）都在
  [`voice_service/protocol.py`](voice_service/protocol.py)，可直接复制（本文件与代码同源）。
