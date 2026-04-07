import { Router, type IRouter, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

const router: IRouter = Router();

// ─── Multi-account client pool with round-robin ───────────────────────────────
//
// Supports two credential sources (can mix and match):
//
// 1. Replit AI Integrations (on Replit):
//    AI_INTEGRATIONS_OPENAI_BASE_URL / AI_INTEGRATIONS_OPENAI_API_KEY
//    AI_INTEGRATIONS_ANTHROPIC_BASE_URL / AI_INTEGRATIONS_ANTHROPIC_API_KEY
//
// 2. Direct API keys (own server or additional accounts):
//    OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3, ...
//    ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, ...
//
// All configured clients are pooled and requests are distributed round-robin
// so multiple accounts share the load automatically.

if (!process.env.PROXY_API_KEY) {
  console.error(
    "[proxy] FATAL: PROXY_API_KEY is not set.\n" +
    "  → On Replit: Tools → Secrets → add PROXY_API_KEY.\n" +
    "  → On your server: set the PROXY_API_KEY environment variable.",
  );
  process.exit(1);
}

const openaiPool: OpenAI[] = [];

// Replit AI Integrations slot
if (process.env.AI_INTEGRATIONS_OPENAI_BASE_URL && process.env.AI_INTEGRATIONS_OPENAI_API_KEY) {
  openaiPool.push(new OpenAI({
    baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
    apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  }));
}

// Direct API keys: OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3, …
for (const varName of ["OPENAI_API_KEY", ...Array.from({ length: 9 }, (_, i) => `OPENAI_API_KEY_${i + 2}`)]) {
  const key = process.env[varName];
  if (key) openaiPool.push(new OpenAI({ apiKey: key }));
}

if (openaiPool.length === 0) {
  console.error(
    "[proxy] FATAL: No OpenAI credentials found.\n" +
    "  → On Replit: Tools → Integrations → enable OpenAI.\n" +
    "  → On your server: set OPENAI_API_KEY (and optionally OPENAI_API_KEY_2, etc.).",
  );
  process.exit(1);
}

const anthropicPool: Anthropic[] = [];

// Replit AI Integrations slot
if (process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL && process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY) {
  anthropicPool.push(new Anthropic({
    baseURL: process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL,
    apiKey: process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY,
  }));
}

// Direct API keys: ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, …
for (const varName of ["ANTHROPIC_API_KEY", ...Array.from({ length: 9 }, (_, i) => `ANTHROPIC_API_KEY_${i + 2}`)]) {
  const key = process.env[varName];
  if (key) anthropicPool.push(new Anthropic({ apiKey: key }));
}

if (anthropicPool.length === 0) {
  console.error(
    "[proxy] FATAL: No Anthropic credentials found.\n" +
    "  → On Replit: Tools → Integrations → enable Anthropic.\n" +
    "  → On your server: set ANTHROPIC_API_KEY (and optionally ANTHROPIC_API_KEY_2, etc.).",
  );
  process.exit(1);
}

console.log(
  `[proxy] Ready — OpenAI: ${openaiPool.length} account(s), Anthropic: ${anthropicPool.length} account(s)`,
);

// Round-robin counter — advances once per incoming request
let rrCounter = 0;
function getClients(): { openai: OpenAI; anthropic: Anthropic } {
  const idx = rrCounter++;
  if (rrCounter >= 1_000_000) rrCounter = 0;
  return {
    openai: openaiPool[idx % openaiPool.length],
    anthropic: anthropicPool[idx % anthropicPool.length],
  };
}

const OPENAI_MODELS = [
  { id: "gpt-5.2", provider: "openai" },
  { id: "gpt-5-mini", provider: "openai" },
  { id: "gpt-5-nano", provider: "openai" },
  { id: "o4-mini", provider: "openai" },
  { id: "o3", provider: "openai" },
];

const ANTHROPIC_MODELS = [
  { id: "claude-opus-4-6", provider: "anthropic" },
  { id: "claude-sonnet-4-6", provider: "anthropic" },
  { id: "claude-haiku-4-5", provider: "anthropic" },
];

const ALL_MODELS = [...OPENAI_MODELS, ...ANTHROPIC_MODELS];

function verifyBearer(req: Request, res: Response): boolean {
  const authHeader = req.headers["authorization"] ?? "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  if (!token || token !== process.env.PROXY_API_KEY) {
    res.status(401).json({ error: { message: "Unauthorized", type: "invalid_request_error", code: "invalid_api_key" } });
    return false;
  }
  return true;
}

function isAnthropicModel(model: string): boolean {
  return model.startsWith("claude-");
}

function isOpenAIModel(model: string): boolean {
  // Match gpt-* and o-series reasoning models (o1, o3, o4-mini, etc.) only.
  // Use a tighter check for "o" models to avoid misrouting unrelated model names.
  return model.startsWith("gpt-") || /^o\d/.test(model);
}

// ─── Tool format conversion helpers ───────────────────────────────────────────

type OpenAITool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
};

type AnthropicTool = {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
};

function openaiToolToAnthropic(tool: OpenAITool): AnthropicTool {
  return {
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters ?? { type: "object", properties: {} },
  };
}

function anthropicToolToOpenAI(tool: AnthropicTool): OpenAITool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  };
}

function openaiToolChoiceToAnthropic(
  tc: unknown,
): Anthropic.MessageCreateParams["tool_choice"] | undefined {
  if (!tc) return undefined;
  if (typeof tc === "string") {
    if (tc === "auto") return { type: "auto" };
    if (tc === "none") return { type: "none" };
    if (tc === "required") return { type: "any" };
  }
  if (typeof tc === "object" && tc !== null) {
    const obj = tc as Record<string, unknown>;
    if (obj.type === "function" && typeof obj.function === "object") {
      const fn = obj.function as Record<string, unknown>;
      return { type: "tool", name: fn.name as string };
    }
  }
  return undefined;
}

function anthropicToolChoiceToOpenAI(
  tc: unknown,
): string | { type: string; function: { name: string } } | undefined {
  if (!tc) return undefined;
  if (typeof tc === "object" && tc !== null) {
    const obj = tc as Record<string, unknown>;
    if (obj.type === "auto") return "auto";
    if (obj.type === "none") return "none";
    if (obj.type === "any") return "required";
    if (obj.type === "tool") return { type: "function", function: { name: obj.name as string } };
  }
  return undefined;
}

// ─── Message conversion helpers ───────────────────────────────────────────────

type AnyMessage = Record<string, unknown>;

function openaiMessagesToAnthropic(
  messages: AnyMessage[],
): { system?: string; messages: Anthropic.MessageParam[] } {
  let system: string | undefined;
  const converted: Anthropic.MessageParam[] = [];

  for (const msg of messages) {
    const role = msg.role as string;
    if (role === "system") {
      system = msg.content as string;
      continue;
    }
    if (role === "user") {
      converted.push({ role: "user", content: msg.content as string });
      continue;
    }
    if (role === "assistant") {
      const toolCalls = msg.tool_calls as Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }> | undefined;
      if (toolCalls && toolCalls.length > 0) {
        const content: Anthropic.ContentBlock[] = [];
        if (msg.content) {
          content.push({ type: "text", text: msg.content as string });
        }
        for (const tc of toolCalls) {
          let input: Record<string, unknown> = {};
          try {
            input = JSON.parse(tc.function.arguments);
          } catch {
            // ignore parse error
          }
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input,
          } as Anthropic.ToolUseBlock);
        }
        converted.push({ role: "assistant", content });
      } else {
        converted.push({ role: "assistant", content: (msg.content as string) ?? "" });
      }
      continue;
    }
    if (role === "tool") {
      const toolUseId = msg.tool_call_id as string;
      const lastMsg = converted[converted.length - 1];
      const toolResultBlock: Anthropic.ToolResultBlockParam = {
        type: "tool_result",
        tool_use_id: toolUseId,
        content: msg.content as string,
      };
      if (lastMsg && lastMsg.role === "user" && Array.isArray(lastMsg.content)) {
        (lastMsg.content as Anthropic.ToolResultBlockParam[]).push(toolResultBlock);
      } else {
        converted.push({ role: "user", content: [toolResultBlock] });
      }
      continue;
    }
  }

  return { system, messages: converted };
}

function anthropicMessagesToOpenAI(messages: AnyMessage[]): AnyMessage[] {
  const converted: AnyMessage[] = [];
  for (const msg of messages) {
    const role = msg.role as string;
    if (role === "user") {
      const content = msg.content;
      if (Array.isArray(content)) {
        // Extract tool_result blocks
        const toolResults = (content as AnyMessage[]).filter(
          (b) => (b as AnyMessage).type === "tool_result",
        );
        const textBlocks = (content as AnyMessage[]).filter(
          (b) => (b as AnyMessage).type !== "tool_result",
        );
        for (const tr of toolResults) {
          converted.push({
            role: "tool",
            tool_call_id: (tr as AnyMessage).tool_use_id,
            content: (tr as AnyMessage).content,
          });
        }
        if (textBlocks.length > 0) {
          const textContent = textBlocks
            .map((b) => ((b as AnyMessage).text as string) ?? "")
            .join("\n");
          converted.push({ role: "user", content: textContent });
        }
      } else {
        converted.push({ role: "user", content });
      }
      continue;
    }
    if (role === "assistant") {
      const content = msg.content;
      if (Array.isArray(content)) {
        const toolUseBlocks = (content as AnyMessage[]).filter(
          (b) => (b as AnyMessage).type === "tool_use",
        );
        const textBlocks = (content as AnyMessage[]).filter(
          (b) => (b as AnyMessage).type === "text",
        );
        const textContent = textBlocks.map((b) => (b as AnyMessage).text).join("\n");
        if (toolUseBlocks.length > 0) {
          converted.push({
            role: "assistant",
            content: textContent || null,
            tool_calls: toolUseBlocks.map((b) => ({
              id: (b as AnyMessage).id,
              type: "function",
              function: {
                name: (b as AnyMessage).name,
                arguments: JSON.stringify((b as AnyMessage).input),
              },
            })),
          });
        } else {
          converted.push({ role: "assistant", content: textContent });
        }
      } else {
        converted.push({ role: "assistant", content });
      }
      continue;
    }
    converted.push(msg);
  }
  return converted;
}

// ─── Prompt caching helper ────────────────────────────────────────────────────

type CacheControl = { type: "ephemeral" };

// Anthropic's minimum cacheable content: 1024 tokens for Haiku, 2048 for others.
// 1 token ≈ 4 chars. Use 1024 tokens (4096 chars) as the universal floor so we
// never pay a cache-write surcharge on content that Anthropic won't actually cache.
const MIN_CACHE_CHARS = 4096;

function blockTextLength(block: Record<string, unknown>): number {
  if (typeof block.text === "string") return block.text.length;
  if (typeof block.thinking === "string") return block.thinking.length;
  if (typeof block.content === "string") return block.content.length;
  return 0;
}

function addCacheControl<T extends Record<string, unknown>>(block: T): T {
  if ((block as Record<string, unknown>).cache_control) return block;
  return { ...block, cache_control: { type: "ephemeral" } as CacheControl };
}

// Anthropic allows at most 4 cache_control markers per request
const MAX_CACHE_POINTS = 4;

function countExistingCachePoints(
  system: string | Anthropic.TextBlockParam[] | undefined,
  messages: Anthropic.MessageParam[],
): number {
  let count = 0;
  if (Array.isArray(system)) {
    for (const block of system) {
      if ((block as Record<string, unknown>).cache_control) count++;
    }
  }
  for (const msg of messages) {
    const content = msg.content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if ((block as Record<string, unknown>).cache_control) count++;
      }
    }
  }
  return count;
}

function applyPromptCaching(
  system: string | Anthropic.TextBlockParam[] | undefined,
  messages: Anthropic.MessageParam[],
): {
  system: Anthropic.TextBlockParam[] | undefined;
  messages: Anthropic.MessageParam[];
} {
  // Count how many cache_control markers the client already sent
  const alreadyUsed = countExistingCachePoints(system, messages);
  let pointsUsed = alreadyUsed;

  // Cache system prompt only if budget allows and it's not already cached
  let cachedSystem: Anthropic.TextBlockParam[] | undefined;
  const systemAlreadyCached =
    Array.isArray(system) && system.some((b) => (b as Record<string, unknown>).cache_control);

  if (!system) {
    cachedSystem = undefined;
  } else if (systemAlreadyCached) {
    // Already has cache_control — pass through unchanged
    cachedSystem = system as Anthropic.TextBlockParam[];
  } else if (pointsUsed < MAX_CACHE_POINTS) {
    const systemText = typeof system === "string" ? system : system.map((b) => b.text).join("");
    if (systemText.length >= MIN_CACHE_CHARS) {
      if (typeof system === "string") {
        cachedSystem = [{ type: "text", text: system, cache_control: { type: "ephemeral" } }];
      } else {
        const arr = [...system];
        arr[arr.length - 1] = addCacheControl(arr[arr.length - 1] as Record<string, unknown>) as Anthropic.TextBlockParam;
        cachedSystem = arr;
      }
      pointsUsed++;
    } else {
      // System prompt too short to be cacheable — don't waste a write surcharge
      cachedSystem = typeof system === "string"
        ? [{ type: "text", text: system }]
        : (system as Anthropic.TextBlockParam[]);
    }
  } else {
    // No budget left — pass system through as array without adding cache_control
    cachedSystem = typeof system === "string"
      ? [{ type: "text", text: system }]
      : (system as Anthropic.TextBlockParam[]);
  }

  // Cache up to remaining budget from message history (skip only the new user turn at the end)
  // messages = [...history, new_user_turn], so candidates are indices 0..length-2
  const remainingPoints = MAX_CACHE_POINTS - pointsUsed;
  const cacheUpTo = Math.max(0, messages.length - 1);

  // Prefer the most recent messages (tail of history) — they're closest to the current turn
  // and most likely to be a reusable cache prefix on the next request.
  const indicesToCache = new Set<number>();
  if (remainingPoints > 0 && cacheUpTo > 0) {
    const count = Math.min(remainingPoints, cacheUpTo);
    for (let k = 0; k < count; k++) {
      indicesToCache.add(cacheUpTo - 1 - k);
    }
  }

  const cachedMessages = messages.map((msg, i) => {
    if (!indicesToCache.has(i)) return msg;
    const content = msg.content;
    if (typeof content === "string") {
      // Only cache if content is long enough to meet Anthropic's minimum threshold
      if (content.length < MIN_CACHE_CHARS) return msg;
      return {
        ...msg,
        content: [addCacheControl({ type: "text" as const, text: content })],
      };
    }
    if (Array.isArray(content) && content.length > 0) {
      const lastBlock = content[content.length - 1] as Record<string, unknown>;
      // Only cache if the last block has enough content to meet the minimum threshold
      if (blockTextLength(lastBlock) < MIN_CACHE_CHARS) return msg;
      const arr = [...content] as Array<Record<string, unknown>>;
      arr[arr.length - 1] = addCacheControl(arr[arr.length - 1]);
      return { ...msg, content: arr as Anthropic.MessageParam["content"] };
    }
    return msg;
  });

  return { system: cachedSystem, messages: cachedMessages };
}

// ─── Strip unsupported cache_control fields ───────────────────────────────────
// Anthropic SDK v0.82+ automatically injects a `scope` field inside
// cache_control objects, but Replit's Anthropic proxy rejects it.
// Strip `scope` (and any other unrecognised keys) before every Anthropic call.

function stripCacheControlScope(block: Record<string, unknown>): Record<string, unknown> {
  if (!block.cache_control || typeof block.cache_control !== "object") return block;
  const { scope: _scope, ...rest } = block.cache_control as Record<string, unknown>;
  // If only scope was present, drop cache_control entirely to avoid sending {}
  if (Object.keys(rest).length === 0) {
    const { cache_control: _cc, ...blockWithout } = block;
    return blockWithout;
  }
  return { ...block, cache_control: rest };
}

function stripScopeFromAnthropicParams(params: Anthropic.MessageCreateParams): Anthropic.MessageCreateParams {
  const system = params.system;
  const cleanSystem = Array.isArray(system)
    ? system.map((b) => stripCacheControlScope(b as Record<string, unknown>) as Anthropic.TextBlockParam)
    : system;

  const cleanMessages = params.messages.map((msg) => {
    const content = msg.content;
    if (typeof content === "string") return msg;
    if (Array.isArray(content)) {
      return {
        ...msg,
        content: content.map((block) =>
          stripCacheControlScope(block as Record<string, unknown>)
        ) as Anthropic.MessageParam["content"],
      };
    }
    return msg;
  });

  // Also strip scope from tool definitions that carry cache_control
  const cleanTools = params.tools
    ? params.tools.map((t) => stripCacheControlScope(t as unknown as Record<string, unknown>) as unknown as Anthropic.Tool)
    : params.tools;

  return {
    ...params,
    ...(cleanSystem !== undefined ? { system: cleanSystem } : {}),
    messages: cleanMessages,
    ...(cleanTools !== undefined ? { tools: cleanTools } : {}),
  };
}

// ─── Response conversion helpers ─────────────────────────────────────────────

function anthropicResponseToOpenAI(
  msg: Anthropic.Message,
  model: string,
): Record<string, unknown> {
  const choices: unknown[] = [];
  const toolCalls: unknown[] = [];
  let textContent = "";
  let thinkingContent = "";

  for (const block of msg.content) {
    if (block.type === "text") {
      textContent += block.text;
    } else if (block.type === "thinking") {
      thinkingContent += (block as unknown as { thinking: string }).thinking;
    } else if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input),
        },
      });
    }
  }

  const finishReason = msg.stop_reason === "tool_use" ? "tool_calls" : "stop";
  const message: Record<string, unknown> = { role: "assistant", content: textContent || null };
  if (thinkingContent) {
    message.reasoning_content = thinkingContent;
  }
  if (toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  choices.push({
    index: 0,
    message,
    finish_reason: finishReason,
  });

  return {
    id: msg.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices,
    usage: {
      prompt_tokens: msg.usage.input_tokens,
      completion_tokens: msg.usage.output_tokens,
      total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
    },
  };
}

function openaiResponseToAnthropic(
  response: OpenAI.ChatCompletion,
  model: string,
): Record<string, unknown> {
  const choice = response.choices[0];
  const content: unknown[] = [];

  if (choice?.message.content) {
    content.push({ type: "text", text: choice.message.content });
  }

  if (choice?.message.tool_calls) {
    for (const tc of choice.message.tool_calls) {
      let input: Record<string, unknown> = {};
      try {
        input = JSON.parse(tc.function.arguments);
      } catch {
        // ignore
      }
      content.push({ type: "tool_use", id: tc.id, name: tc.function.name, input });
    }
  }

  const stopReason =
    choice?.finish_reason === "tool_calls" ? "tool_use" : "end_turn";

  return {
    id: response.id,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
    },
  };
}

// ─── GET /v1/models ───────────────────────────────────────────────────────────

router.get("/models", (req, res) => {
  if (!verifyBearer(req, res)) return;
  res.json({
    object: "list",
    data: ALL_MODELS.map((m) => ({
      id: m.id,
      object: "model",
      created: 1700000000,
      owned_by: m.provider,
    })),
  });
});

// ─── POST /v1/chat/completions ────────────────────────────────────────────────

router.post("/chat/completions", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;
  const { openai, anthropic } = getClients();

  const body = req.body as Record<string, unknown>;
  const model = (body.model as string) ?? "gpt-5.2";
  const stream = body.stream === true;
  const messages = (body.messages as AnyMessage[]) ?? [];
  const tools = body.tools as OpenAITool[] | undefined;
  const toolChoice = body.tool_choice;
  const maxTokens = (body.max_tokens as number | undefined) ?? 8192;

  try {
    // ── OpenAI path ──
    if (isOpenAIModel(model)) {
      const params: OpenAI.ChatCompletionCreateParams = {
        model,
        messages: messages as OpenAI.ChatCompletionMessageParam[],
        stream,
        ...(tools ? { tools } : {}),
        ...(toolChoice !== undefined ? { tool_choice: toolChoice as OpenAI.ChatCompletionToolChoiceOption } : {}),
        ...(body.temperature !== undefined ? { temperature: body.temperature as number } : {}),
        ...(body.top_p !== undefined ? { top_p: body.top_p as number } : {}),
        max_completion_tokens: maxTokens,
      };

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          res.write(": keepalive\n\n");
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        }, 5000);

        try {
          const oaiStream = await openai.chat.completions.create({ ...params, stream: true });
          for await (const chunk of oaiStream) {
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
              (res as unknown as { flush: () => void }).flush();
            }
          }
          res.write("data: [DONE]\n\n");
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream OpenAI
      const response = await openai.chat.completions.create({ ...params, stream: false });
      res.json(response);
      return;
    }

    // ── Anthropic path ──
    if (isAnthropicModel(model)) {
      const { system: rawSystem, messages: rawAnthropicMessages } = openaiMessagesToAnthropic(messages);
      const anthropicTools = tools ? tools.map(openaiToolToAnthropic) : undefined;
      const anthropicToolChoice = openaiToolChoiceToAnthropic(toolChoice);

      // Auto prompt caching: reduces token cost for repeated context
      const { system: cachedSystem, messages: anthropicMessages } = applyPromptCaching(rawSystem, rawAnthropicMessages);

      const thinking = body.thinking as Anthropic.MessageCreateParams["thinking"] | undefined;
      const anthropicParams: Anthropic.MessageCreateParams = stripScopeFromAnthropicParams({
        model,
        max_tokens: maxTokens,
        messages: anthropicMessages,
        ...(cachedSystem ? { system: cachedSystem } : {}),
        ...(anthropicTools ? { tools: anthropicTools } : {}),
        ...(anthropicToolChoice ? { tool_choice: anthropicToolChoice } : {}),
        ...(thinking ? { thinking } : {}),
      });

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          res.write(": keepalive\n\n");
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        }, 5000);

        // State for building streaming OpenAI-format chunks
        let messageId = `chatcmpl-${Date.now()}`;
        let currentBlockIndex = -1;
        let currentBlockType = "";
        let currentToolId = "";
        let currentToolName = "";

        const flush = () => {
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        };

        const sendChunk = (delta: Record<string, unknown>, finishReason: string | null = null) => {
          const chunk = {
            id: messageId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [{ index: 0, delta, finish_reason: finishReason }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          flush();
        };

        try {
          const anthropicStream = anthropic.messages.stream(anthropicParams);

          for await (const event of anthropicStream) {
            if (event.type === "message_start") {
              messageId = event.message.id || messageId;
              sendChunk({ role: "assistant" });
            } else if (event.type === "content_block_start") {
              currentBlockIndex = event.index;
              const block = event.content_block;
              if (block.type === "text") {
                currentBlockType = "text";
              } else if (block.type === "thinking") {
                currentBlockType = "thinking";
              } else if (block.type === "tool_use") {
                currentBlockType = "tool_use";
                currentToolId = block.id;
                currentToolName = block.name;
                sendChunk({
                  tool_calls: [{
                    index: currentBlockIndex,
                    id: currentToolId,
                    type: "function",
                    function: { name: currentToolName, arguments: "" },
                  }],
                });
              }
            } else if (event.type === "content_block_delta") {
              const delta = event.delta;
              if (delta.type === "text_delta") {
                sendChunk({ content: delta.text });
              } else if (delta.type === "thinking_delta") {
                sendChunk({ reasoning_content: (delta as unknown as { thinking: string }).thinking });
              } else if (delta.type === "input_json_delta") {
                sendChunk({
                  tool_calls: [{
                    index: currentBlockIndex,
                    function: { arguments: delta.partial_json },
                  }],
                });
              }
            } else if (event.type === "message_delta") {
              const stopReason =
                event.delta.stop_reason === "tool_use" ? "tool_calls" : "stop";
              const usageChunk: Record<string, unknown> = {};
              if (event.usage) {
                usageChunk.usage = {
                  prompt_tokens: 0,
                  completion_tokens: event.usage.output_tokens ?? 0,
                  total_tokens: event.usage.output_tokens ?? 0,
                };
              }
              sendChunk(usageChunk, stopReason);
            } else if (event.type === "message_stop") {
              res.write("data: [DONE]\n\n");
              flush();
            }
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream Anthropic — use stream().finalMessage() to avoid timeouts
      const finalMsg = await anthropic.messages.stream(anthropicParams).finalMessage();
      res.json(anthropicResponseToOpenAI(finalMsg, model));
      return;
    }

    res.status(400).json({ error: { message: `Unknown model: ${model}`, type: "invalid_request_error" } });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    req.log?.error({ err }, "proxy error");
    if (!res.headersSent) {
      res.status(500).json({ error: { message: errMsg, type: "api_error" } });
    } else {
      res.end();
    }
  }
});

// ─── POST /v1/messages (Anthropic native) ────────────────────────────────────

router.post("/messages", async (req: Request, res: Response) => {
  if (!verifyBearer(req, res)) return;
  const { openai, anthropic } = getClients();

  const body = req.body as Record<string, unknown>;
  const model = (body.model as string) ?? "claude-sonnet-4-6";
  const stream = body.stream === true;
  const messages = (body.messages as AnyMessage[]) ?? [];
  const system = body.system as string | Anthropic.TextBlockParam[] | undefined;
  const tools = body.tools as AnthropicTool[] | undefined;
  const toolChoice = body.tool_choice;
  const maxTokens = (body.max_tokens as number | undefined) ?? 8192;

  try {
    // ── Claude model → direct Anthropic ──
    if (isAnthropicModel(model)) {
      const thinking = body.thinking as Anthropic.MessageCreateParams["thinking"] | undefined;

      // Auto prompt caching: reduces token cost for repeated context
      const { system: cachedSystem, messages: cachedMessages } = applyPromptCaching(
        system,
        messages as Anthropic.MessageParam[],
      );

      const params: Anthropic.MessageCreateParams = stripScopeFromAnthropicParams({
        model,
        max_tokens: maxTokens,
        messages: cachedMessages,
        ...(cachedSystem ? { system: cachedSystem } : {}),
        ...(tools ? { tools } : {}),
        ...(toolChoice ? { tool_choice: toolChoice as Anthropic.MessageCreateParams["tool_choice"] } : {}),
        ...(thinking ? { thinking } : {}),
      });

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          res.write(": keepalive\n\n");
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        }, 5000);

        const flush = () => {
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        };

        try {
          const anthropicStream = anthropic.messages.stream(params);
          for await (const event of anthropicStream) {
            res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
            flush();
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream
      const msg = await anthropic.messages.stream(params).finalMessage();
      res.json(msg);
      return;
    }

    // ── OpenAI model → convert and proxy ──
    if (isOpenAIModel(model)) {
      const openaiMessages = anthropicMessagesToOpenAI(messages);
      if (system) {
        // OpenAI system messages require a plain string, not an array of content blocks
        const systemContent = Array.isArray(system)
          ? system.map((b) => b.text).join("\n\n")
          : system;
        openaiMessages.unshift({ role: "system", content: systemContent });
      }
      const openaiTools = tools ? tools.map(anthropicToolToOpenAI) : undefined;
      const openaiToolChoice = anthropicToolChoiceToOpenAI(toolChoice);

      const params: OpenAI.ChatCompletionCreateParams = {
        model,
        messages: openaiMessages as OpenAI.ChatCompletionMessageParam[],
        stream,
        max_completion_tokens: maxTokens,
        ...(openaiTools ? { tools: openaiTools } : {}),
        ...(openaiToolChoice !== undefined ? { tool_choice: openaiToolChoice as OpenAI.ChatCompletionToolChoiceOption } : {}),
      };

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        res.flushHeaders();

        const keepalive = setInterval(() => {
          res.write(": keepalive\n\n");
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        }, 5000);

        const flush = () => {
          if ("flush" in res && typeof (res as unknown as { flush: () => void }).flush === "function") {
            (res as unknown as { flush: () => void }).flush();
          }
        };

        let msgId = `msg_${Date.now()}`;
        let inputTokens = 0;
        let outputTokens = 0;
        let sentMessageStart = false;
        let currentIndex = -1;
        let currentToolId = "";
        let currentToolName = "";
        let currentToolInput = "";

        const sendEvent = (type: string, data: Record<string, unknown>) => {
          res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
          flush();
        };

        try {
          const oaiStream = await openai.chat.completions.create({ ...params, stream: true });

          for await (const chunk of oaiStream) {
            if (!sentMessageStart) {
              msgId = chunk.id || msgId;
              sentMessageStart = true;
              sendEvent("message_start", {
                type: "message_start",
                message: {
                  id: msgId,
                  type: "message",
                  role: "assistant",
                  content: [],
                  model,
                  stop_reason: null,
                  stop_sequence: null,
                  usage: { input_tokens: 0, output_tokens: 0 },
                },
              });
              sendEvent("ping", { type: "ping" });
            }

            const delta = chunk.choices[0]?.delta;
            const finishReason = chunk.choices[0]?.finish_reason;

            if (delta?.content) {
              if (currentIndex < 0) {
                currentIndex = 0;
                sendEvent("content_block_start", {
                  type: "content_block_start",
                  index: currentIndex,
                  content_block: { type: "text", text: "" },
                });
              }
              sendEvent("content_block_delta", {
                type: "content_block_delta",
                index: currentIndex,
                delta: { type: "text_delta", text: delta.content },
              });
              outputTokens += 1;
            }

            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                const tcIndex = tc.index ?? 0;
                if (tc.id) {
                  // new tool call block
                  if (currentIndex >= 0 && currentToolId) {
                    sendEvent("content_block_stop", { type: "content_block_stop", index: currentIndex });
                  }
                  currentIndex = tcIndex + 1;
                  currentToolId = tc.id;
                  currentToolName = tc.function?.name ?? "";
                  currentToolInput = "";
                  sendEvent("content_block_start", {
                    type: "content_block_start",
                    index: currentIndex,
                    content_block: { type: "tool_use", id: currentToolId, name: currentToolName, input: {} },
                  });
                }
                if (tc.function?.arguments) {
                  currentToolInput += tc.function.arguments;
                  sendEvent("content_block_delta", {
                    type: "content_block_delta",
                    index: currentIndex,
                    delta: { type: "input_json_delta", partial_json: tc.function.arguments },
                  });
                }
              }
            }

            if (finishReason) {
              if (currentIndex >= 0) {
                sendEvent("content_block_stop", { type: "content_block_stop", index: currentIndex });
              }
              const stopReason = finishReason === "tool_calls" ? "tool_use" : "end_turn";
              sendEvent("message_delta", {
                type: "message_delta",
                delta: { stop_reason: stopReason, stop_sequence: null },
                usage: { output_tokens: outputTokens },
              });
              sendEvent("message_stop", { type: "message_stop" });
            }

            if (chunk.usage) {
              inputTokens = chunk.usage.prompt_tokens ?? 0;
              outputTokens = chunk.usage.completion_tokens ?? 0;
            }
          }
        } finally {
          clearInterval(keepalive);
          res.end();
        }
        return;
      }

      // Non-stream: OpenAI → Anthropic format
      const response = await openai.chat.completions.create({ ...params, stream: false });
      res.json(openaiResponseToAnthropic(response, model));
      return;
    }

    res.status(400).json({ error: { message: `Unknown model: ${model}`, type: "invalid_request_error" } });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    req.log?.error({ err }, "proxy error");
    if (!res.headersSent) {
      res.status(500).json({ error: { message: errMsg, type: "api_error" } });
    } else {
      res.end();
    }
  }
});

export default router;
