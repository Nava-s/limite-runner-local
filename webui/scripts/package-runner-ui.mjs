import {copyFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
const out = resolve(process.env.LLAMA_UI_OUT_DIR || '../src/limite_runner/server/static/llama-ui');
copyFileSync('LICENSE.upstream', resolve(out,'LICENSE.llama-ui.txt'));
writeFileSync(resolve(out,'UPSTREAM.txt'), 'llama.cpp tools/ui @ 58367713a6935c0810103378144008df32e3d5db\nAdapted for Limite Runner; see webui/README.md in the source repository.\n');
