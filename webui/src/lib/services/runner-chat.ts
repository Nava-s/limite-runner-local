/** Limite transport: local history is presentation only; never send tools or overrides. */
export function runnerRequest(messages: {role: string; content: unknown}[], options: Record<string, unknown> = {}) {
 const question = messages.findLast(m => m.role === 'user');
 if (!question || typeof question.content !== 'string' || !question.content.trim())
  throw new Error('Limite richiede una domanda di testo.');
 const body: Record<string, unknown> = {messages: [{role: 'user', content: question.content}], stream: true};
 for (const key of ['temperature', 'top_p', 'max_tokens']) {
  const value = options[key];
  if (typeof value === 'number' && Number.isFinite(value) && (key !== 'max_tokens' || value > 0)) body[key] = value;
 }
 return body;
}

export async function sendRunnerMessage(
 messages: {role: string; content: unknown}[], options: SettingsChatServiceOptions, signal?: AbortSignal
): Promise<void> {
 let raw = '', text = '', reasoning = '', finish: string | null = null;
 // Keep incomplete tags buffered: token boundaries can split <think> and </think>.
 function publish(final = false) {
  let nextText = raw, nextReasoning = '';
  const start = raw.indexOf('<think>');
  if (start >= 0) {
   const end = raw.indexOf('</think>', start + 7);
   nextReasoning = raw.slice(start + 7, end < 0 ? undefined : end);
   nextText = raw.slice(0, start) + (end < 0 ? '' : raw.slice(end + 8));
  }
  if (!final) {
   nextText = nextText.replace(/<[^>]*$/, '');
   nextReasoning = nextReasoning.replace(/<[^>]*$/, '');
  }
  if (nextReasoning.length > reasoning.length) options.onReasoningChunk?.(nextReasoning.slice(reasoning.length));
  if (nextText.length > text.length) options.onChunk?.(nextText.slice(text.length));
  reasoning = nextReasoning; text = nextText;
 }
 function conclude(label: string) {
  publish(true);
  const notice = '\n\n---\n*' + label + '*';
  text += notice;
  options.onChunk?.(notice);
  options.onComplete?.(text, reasoning || undefined);
 }
 try {
  const response = await fetch('/v1/chat/completions', {
   method: 'POST', headers: {'Content-Type': 'application/json'},
   body: JSON.stringify(runnerRequest(messages, options as Record<string, unknown>)), signal
  });
  if (!response.ok) {
   const error = await response.json().catch(() => ({}));
   throw new Error(error.detail || error.error?.message || `HTTP ${response.status}`);
  }
  if (!response.body) throw new Error('Stream assente: risposta non verificata.');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '';
  const consume = (line: string) => {
   if (!line.startsWith('data:')) return;
   const value = line.slice(5).trim();
   if (!value || value === '[DONE]') return;
   const event = JSON.parse(value);
   if (event.error !== undefined) throw new Error(typeof event.error === 'string' ? event.error : event.error.message || 'Errore server');
   if (event.model) options.onModel?.(event.model);
   const choice = event.choices?.[0];
   if (choice?.delta?.content) { raw += choice.delta.content; publish(); }
   if (choice?.finish_reason) finish = choice.finish_reason;
  };
  try {
   while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value, {stream: !done});
    const lines = buffer.split('\n'); buffer = lines.pop() || '';
    for (const line of lines) consume(line.trimEnd());
    if (done) { if (buffer) consume(buffer); break; }
   }
  } finally { reader.releaseLock(); }
  if (finish === 'stop') conclude('Risposta conclusa');
  else if (finish === 'length') conclude('Limite di token raggiunto: risposta troncata');
  else throw new Error('Risposta NON verificata come completa (stato finale assente o sconosciuto).');
 } catch (error) {
  if (signal?.aborted) { conclude('Interrotta'); return; }
  const failure = error instanceof Error ? error : new Error(String(error));
  options.onError?.(failure);
  throw failure;
 }
}
