import test from 'node:test';
import assert from 'node:assert/strict';
import {runnerRequest, sendRunnerMessage} from './src/lib/services/runner-chat.ts';

test('only last user text crosses the transport boundary', () => {
 const request = runnerRequest([{role:'system',content:'override'}, {role:'user',content:'first'}, {role:'assistant',content:'old'}, {role:'user',content:'second'}], {tools:[{}],custom:{messages:[]},temperature:0.5,max_tokens:64});
 assert.deepEqual(request,{messages:[{role:'user',content:'second'}],stream:true,temperature:0.5,max_tokens:64});
});
test('unset/legacy infinite cap uses server default', () => {
 assert.equal(runnerRequest([{role:'user',content:'hello'}],{max_tokens:-1}).max_tokens,undefined);
});
test('multimodal messages fail clearly', () => {
 assert.throws(()=>runnerRequest([{role:'user',content:[{type:'image_url'}]}]),/testo/);
});
async function simulate(events) {
 let text='', reason='', complete=false, failure;
 const old=globalThis.fetch;
 globalThis.fetch=async()=>new Response(new ReadableStream({start(c){
  // Deliberately split UTF-8 and SSE boundaries one byte at a time.
  for(const byte of new TextEncoder().encode(events)) c.enqueue(new Uint8Array([byte])); c.close();
 }}));
 try {await sendRunnerMessage([{role:'user',content:'test'}],{onChunk:c=>text+=c,onReasoningChunk:c=>reason+=c,onComplete:()=>complete=true,onError:e=>failure=e});}
 catch(e){failure=e;} finally{globalThis.fetch=old;}
 return {text,reason,complete,failure};
}
const chunk=(content,finish_reason=null)=>'data: '+JSON.stringify({choices:[{delta:{content},finish_reason}]})+'\r\n\r\n';
test('split reasoning tags and UTF-8 are preserved and collapsed separately',async()=>{
 const result=await simulate(chunk('<thi')+chunk('nk>perché')+chunk('</thi')+chunk('nk>4')+chunk('', 'stop')+'data: [DONE]\n\n');
 assert.equal(result.reason,'perché');assert.match(result.text,/^4/);assert.match(result.text,/Risposta conclusa/);assert.equal(result.complete,true);
});
test('length is visibly truncated',async()=>{const r=await simulate(chunk('partial','length'));assert.match(r.text,/troncata/);assert.equal(r.complete,true);});
test('EOF without final reason is an error',async()=>{const r=await simulate(chunk('partial'));assert.equal(r.complete,false);assert.match(r.failure.message,/NON verificata/);});
test('SSE error is not silently accepted',async()=>{const r=await simulate('data: {"error":"backend stopped"}\n\n');assert.equal(r.complete,false);assert.match(r.failure.message,/backend stopped/);});
test('abort marks output as interrupted',async()=>{
 const controller=new AbortController();controller.abort();const old=globalThis.fetch;let text='';
 globalThis.fetch=async()=>{throw new DOMException('aborted','AbortError');};
 try{await sendRunnerMessage([{role:'user',content:'test'}],{onChunk:c=>text+=c},controller.signal);}finally{globalThis.fetch=old;}
 assert.match(text,/Interrotta/);assert.doesNotMatch(text,/conclusa/);
});
