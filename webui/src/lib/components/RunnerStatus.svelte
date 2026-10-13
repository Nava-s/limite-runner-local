<script lang="ts">
 import {onMount} from 'svelte';
 let status = $state('verifica connessione…');
 onMount(() => {
  let disposed = false;
  const controller = new AbortController();
  async function check() {
   try {
    const response = await fetch('/health', {cache:'no-store', signal:controller.signal});
    const body = await response.json();
    if (!disposed) status = body.service !== 'limite-runner' ? 'servizio errato' : body.ready ? 'pronto' : 'motore non pronto';
   } catch {if (!disposed) status = 'non raggiungibile';}
  }
  void check();
  const timer = setInterval(check, 5000);
  return () => {disposed=true;clearInterval(timer);controller.abort();};
 });
</script>
<p class="mx-auto max-w-3xl px-4 pt-2 text-center text-xs text-muted-foreground" aria-live="polite">
 Limite Runner · {status} · Ogni domanda è indipendente. La cronologia resta in questa chat.
</p>
