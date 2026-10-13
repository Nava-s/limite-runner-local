import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { mdsvex } from 'mdsvex';

// CMake sets LLAMA_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` writes the assets shipped by the Python package.
const outDir = process.env.LLAMA_UI_OUT_DIR ?? '../src/limite_runner/server/static/llama-ui';
/** @type {import('@sveltejs/kit').Config} */
const config = {
	extensions: ['.svelte', '.svx'],

	kit: {
		version: { name: 'limite-runner-0.1.4-ui1' },
		adapter: adapter({
			assets: outDir,
			fallback: 'index.html',
			pages: outDir,
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'single'
		},
		paths: {
			relative: true
		},
		router: { type: 'hash' }
	},

	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()]
};

export default config;
