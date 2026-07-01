/**
 * Pins `toModelMessages` — the gbrain ChatMessage[] → AI SDK v6 ModelMessage[]
 * converter. v6 tightened ModelMessage validation: tool results must ride on a
 * dedicated `role:'tool'` message with a structured `{type,value}` output part,
 * not a `role:'user'` message with a bare-value tool-result block (which is how
 * gbrain's toolLoop pushes them). Without this conversion every multi-turn tool
 * loop — skillopt rollouts AND production subagent jobs — throws "messages do
 * not match the ModelMessage[] schema" the moment the model calls a tool.
 *
 * Surfaced by the SkillOpt real-LLM eval (Track B). These cases pin the exact
 * v6 shapes that `generateText` accepts (verified against AI SDK 6.0.174).
 */
import { describe, test, expect } from 'bun:test';
import { z } from 'zod';
import { modelMessageSchema } from 'ai';
import { toModelMessages, toJsonSafeValue, jsonSafeStringify, type ChatMessage } from '../src/core/ai/gateway.ts';

describe('toModelMessages — v6 ModelMessage shape', () => {
  test('string content passes through unchanged', () => {
    const msgs: ChatMessage[] = [{ role: 'user', content: 'hello' }];
    expect(toModelMessages(msgs)).toEqual([{ role: 'user', content: 'hello' }]);
  });

  test('assistant text block maps to {type:text,text}', () => {
    const msgs: ChatMessage[] = [
      { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
    ];
    expect(toModelMessages(msgs)).toEqual([
      { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
    ]);
  });

  test('assistant tool-call block keeps {toolCallId,toolName,input}', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'assistant',
        content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: { query: 'x' } }],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      {
        role: 'assistant',
        content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: { query: 'x' } }],
      },
    ]);
  });

  test('tool-result on a user-role message becomes role:tool with json output', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'user',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { hits: 0 } }],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { type: 'json', value: { hits: 0 } } },
        ],
      },
    ]);
  });

  test('string tool-result output becomes {type:text,value}', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'user',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'echo', output: 'done' }],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      {
        role: 'tool',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'echo', output: { type: 'text', value: 'done' } }],
      },
    ]);
  });

  test('errored tool-result becomes {type:error-text,value}', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'user',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { msg: 'boom' }, isError: true }],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      {
        role: 'tool',
        content: [
          { type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { type: 'error-text', value: '{"msg":"boom"}' } },
        ],
      },
    ]);
  });

  test('null tool-result output is preserved as json null (not dropped)', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'user',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'noop', output: null }],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      {
        role: 'tool',
        content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'noop', output: { type: 'json', value: null } }],
      },
    ]);
  });

  // Regression: a reasoning model (e.g. DeepSeek v4) can emit a text part whose
  // `text` is undefined/null. AI SDK v6 rejects it ("messages do not match the
  // ModelMessage[] schema"), failing the whole turn. toModelMessages must drop
  // such parts (mirroring the replay path), so the bug shows up inline but a
  // resumed run — rebuilt via adaptContentBlocksToChatBlocks — recovers.
  test('non-string text block is dropped (v6 rejects undefined/null text)', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'assistant',
        content: [
          { type: 'text', text: undefined as unknown as string },
          { type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: {} },
        ],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: {} }] },
    ]);
  });

  test('valid text alongside a non-string text block is kept; null text dropped', () => {
    const msgs: ChatMessage[] = [
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'real answer' },
          { type: 'text', text: null as unknown as string },
        ],
      },
    ];
    expect(toModelMessages(msgs)).toEqual([
      { role: 'assistant', content: [{ type: 'text', text: 'real answer' }] },
    ]);
  });

  test('empty-string text is kept (valid for v6)', () => {
    const msgs: ChatMessage[] = [{ role: 'assistant', content: [{ type: 'text', text: '' }] }];
    expect(toModelMessages(msgs)).toEqual([{ role: 'assistant', content: [{ type: 'text', text: '' }] }]);
  });

  test('full multi-turn conversation: user → assistant(tool-call) → tool(result)', () => {
    const msgs: ChatMessage[] = [
      { role: 'user', content: 'find widget' },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: { query: 'widget' } }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { hits: 0 } }] },
    ];
    const out = toModelMessages(msgs);
    expect(out).toHaveLength(3);
    expect((out[0] as any).role).toBe('user');
    expect((out[1] as any).role).toBe('assistant');
    expect((out[2] as any).role).toBe('tool');
    expect((out[2] as any).content[0].output).toEqual({ type: 'json', value: { hits: 0 } });
  });

  // 补丁4 (2026-07-01, job 54 实跑抓到): live tool 输出(如 brain_get_page 返回的
  // page 对象)可含非-JSON 值(undefined 字段 / Date / bigint)。toModelMessages 若
  // 原样塞进 {type:'json', value}，AI SDK v6 的 JSONValue schema 拒(invalid_union
  // at ['output','value'])→ 整个 synthesize turn 死。持久化版过了 PG JSONB 往返被
  // 净化，故只有 live 路径炸(9/37 job 死信)。修:tool-result output value 做 JSON 净化。
  test('补丁4: tool-result output 含非-JSON 值(undefined/Date/bigint)应转成合法 ModelMessage[]', () => {
    const msgs: ChatMessage[] = [
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'brain_get_page', input: {} }] },
      {
        role: 'user',
        content: [{
          type: 'tool-result', toolCallId: 'c1', toolName: 'brain_get_page',
          output: {
            id: 43, opt: undefined, when: new Date('2026-06-29T00:00:00Z'),
            big: BigInt('9007199254740993'), nested: { x: undefined, arr: [1, undefined, 3] },
          },
        }],
      },
    ];
    const out = toModelMessages(msgs as any) as any[];
    // 1) 整批过 v6 ModelMessage schema(修前 false: invalid_union at output.value)
    expect(z.array(modelMessageSchema).safeParse(out).success).toBe(true);
    // 2) 具体净化语义(codex P2): Date→ISO / bigint→字符串 / undefined 剥 / 数组 undefined→null
    const val = out[1].content[0].output.value;
    expect(val.when).toBe('2026-06-29T00:00:00.000Z'); // Date → ISO 字符串
    expect(val.big).toBe('9007199254740993');          // bigint → 字符串
    expect('opt' in val).toBe(false);                  // 顶层 undefined 字段被剥
    expect('x' in val.nested).toBe(false);             // 嵌套 undefined 被剥
    expect(val.nested.arr).toEqual([1, null, 3]);      // 数组里 undefined → null
    expect(val.id).toBe(43);                           // 正常值保留
  });
});

// 补丁4: 净化 helper 单测(持久化 choke persistToolExecComplete + toModelMessages 共用)。
describe('补丁4: jsonSafe helpers', () => {
  const circular: any = { a: 1 };
  circular.self = circular;

  test('jsonSafeStringify 永不抛(bigint/循环/Date/undefined)', () => {
    expect(() => jsonSafeStringify({ b: BigInt('9007199254740993') })).not.toThrow();
    expect(() => jsonSafeStringify(circular)).not.toThrow();
    expect(JSON.parse(jsonSafeStringify({ b: BigInt(42) })).b).toBe('42');   // bigint→字符串
    expect(JSON.parse(jsonSafeStringify({ when: new Date('2026-06-29T00:00:00Z') })).when)
      .toBe('2026-06-29T00:00:00.000Z');                                     // Date→ISO
    expect('opt' in JSON.parse(jsonSafeStringify({ opt: undefined, keep: 1 }))).toBe(false); // undefined 剥
    expect(jsonSafeStringify(circular)).toContain('[Circular]');             // 循环→标记,不抛
  });

  test('toJsonSafeValue 净化到合法 JSONValue,永不抛', () => {
    expect(toJsonSafeValue(null)).toBe(null);
    expect(toJsonSafeValue({ big: BigInt(7) })).toEqual({ big: '7' });
    expect(() => toJsonSafeValue(circular)).not.toThrow();
    expect((toJsonSafeValue(circular) as any).self).toBe('[Circular]');
  });
});
