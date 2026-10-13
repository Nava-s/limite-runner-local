// limite-logits: dump greedy-trajectory logits for `limite validate`.
//
// Built against the patched llama.cpp (Limite adapter) by
// scripts/build_llamacpp.sh. Uses only the public llama.cpp C API:
//
//   limite-logits -m model.gguf -c 16384 -n 16 --out logits.npz --prompt "..."
//
// Output is a .npz (stored zip) with:
//   logits          float32 [n, vocab]  greedy-trajectory logits (cached decode)
//   tokens          int32   [n]         greedy token ids (cached decode)
//   tokens_nocache  int32   [n]         greedy token ids, full recompute each step
//
// The no-cache trajectory validates decode coherence with/without KV cache.

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// -- minimal CRC32 (IEEE) ---------------------------------------------------
static uint32_t crc32_table[256];
static void crc32_init() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
}
static uint32_t crc32_of(const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) c = crc32_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// -- minimal .npy v1.0 buffer ------------------------------------------------
static std::vector<uint8_t> npy_buffer(const std::string & descr,
                                       const std::string & shape,
                                       const void * data, size_t nbytes) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    size_t pad = 64 - ((10 + header.size() + 1) % 64);
    header.append(pad, ' ');
    header += '\n';
    std::vector<uint8_t> out;
    out.reserve(10 + header.size() + nbytes);
    out.push_back(0x93); out.insert(out.end(), {'N','U','M','P','Y'});
    out.push_back(1); out.push_back(0);
    uint16_t hlen = (uint16_t) header.size();
    out.push_back(hlen & 0xFF); out.push_back((hlen >> 8) & 0xFF);
    out.insert(out.end(), header.begin(), header.end());
    const uint8_t * p = (const uint8_t *) data;
    out.insert(out.end(), p, p + nbytes);
    return out;
}

// -- minimal stored-zip (.npz) writer ----------------------------------------
struct ZipEntry { std::string name; std::vector<uint8_t> data; };
static void put16(std::vector<uint8_t> & v, uint16_t x) { v.push_back(x & 0xFF); v.push_back((x >> 8) & 0xFF); }
static void put32(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back(x & 0xFF); v.push_back((x >> 8) & 0xFF);
    v.push_back((x >> 16) & 0xFF); v.push_back((x >> 24) & 0xFF);
}
static bool write_npz(const std::string & path, const std::vector<ZipEntry> & entries) {
    std::vector<uint8_t> out;
    std::vector<uint32_t> offsets;
    for (const auto & e : entries) {
        offsets.push_back((uint32_t) out.size());
        uint32_t crc = crc32_of(e.data.data(), e.data.size());
        put32(out, 0x04034b50); put16(out, 20); put16(out, 0); put16(out, 0);
        put16(out, 0); put16(out, 0); put32(out, crc);
        put32(out, (uint32_t) e.data.size()); put32(out, (uint32_t) e.data.size());
        put16(out, (uint16_t) e.name.size()); put16(out, 0);
        out.insert(out.end(), e.name.begin(), e.name.end());
        out.insert(out.end(), e.data.begin(), e.data.end());
    }
    uint32_t cdir_off = (uint32_t) out.size();
    uint32_t cdir_size = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        const auto & e = entries[i];
        uint32_t crc = crc32_of(e.data.data(), e.data.size());
        size_t start = out.size();
        put32(out, 0x02014b50); put16(out, 20); put16(out, 20);
        put16(out, 0); put16(out, 0); put16(out, 0); put16(out, 0);
        put32(out, crc);
        put32(out, (uint32_t) e.data.size()); put32(out, (uint32_t) e.data.size());
        put16(out, (uint16_t) e.name.size());
        put16(out, 0); put16(out, 0); put16(out, 0); put16(out, 0);
        put32(out, 0); put32(out, offsets[i]);
        out.insert(out.end(), e.name.begin(), e.name.end());
        cdir_size += (uint32_t)(out.size() - start);
    }
    put32(out, 0x06054b50); put16(out, 0); put16(out, 0);
    put16(out, (uint16_t) entries.size()); put16(out, (uint16_t) entries.size());
    put32(out, cdir_size); put32(out, cdir_off); put16(out, 0);
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return w == out.size();
}

// -- argmax -------------------------------------------------------------------
static int argmax_f32(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m model.gguf [-c 16384] [-n 16] [--prompt TEXT | --prompt-file F] [--forced IDS.bin] --out logits.npz\n",
        prog);
}

int main(int argc, char ** argv) {
    crc32_init();
    std::string model_path, prompt, prompt_file, out_path, forced_path;
    int n_ctx = 16384, n_gen = 16;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](std::string & dst) {
            if (++i >= argc) { usage(argv[0]); return false; }
            dst = argv[i]; return true;
        };
        if (a == "-m") { if (!need(model_path)) return 2; }
        else if (a == "-c") { std::string s; if (!need(s)) return 2; n_ctx = atoi(s.c_str()); }
        else if (a == "-n") { std::string s; if (!need(s)) return 2; n_gen = atoi(s.c_str()); }
        else if (a == "--prompt") { if (!need(prompt)) return 2; }
        else if (a == "--prompt-file") { if (!need(prompt_file)) return 2; }
        else if (a == "--out") { if (!need(out_path)) return 2; }
        else if (a == "--forced") { if (!need(forced_path)) return 2; }
        else { usage(argv[0]); return 2; }
    }
    if (model_path.empty() || out_path.empty()) { usage(argv[0]); return 2; }
    if (!prompt_file.empty()) {
        FILE * f = fopen(prompt_file.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open prompt file\n"); return 1; }
        char buf[1 << 20]; size_t r = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f); buf[r] = 0; prompt.assign(buf, r);
    }
    if (prompt.empty()) { fprintf(stderr, "empty prompt\n"); return 2; }

    // Teacher forcing: token ids (int32 little-endian, raw) fed as the input
    // of each generation step instead of the candidate's own argmax. The
    // logits recorded at step s are then conditioned on forced[0..s-1].
    std::vector<llama_token> forced;
    if (!forced_path.empty()) {
        FILE * f = fopen(forced_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open --forced file\n"); return 1; }
        std::vector<int32_t> raw;
        int32_t v;
        while (fread(&v, sizeof(v), 1, f) == 1) raw.push_back(v);
        fclose(f);
        if ((int) raw.size() < n_gen) {
            fprintf(stderr, "--forced has %zu ids, need %d\n", raw.size(), n_gen);
            return 2;
        }
        forced.assign(raw.begin(), raw.begin() + n_gen);
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // tokenize: template already contains BOS explicitly -> parse specials, no add
    std::vector<llama_token> prompt_tokens(prompt.size() + 16);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                  prompt_tokens.data(), (int32_t) prompt_tokens.size(),
                                  false, true);
    if (n_prompt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    prompt_tokens.resize(n_prompt);
    if (n_prompt + n_gen > n_ctx) { fprintf(stderr, "prompt + n exceeds ctx\n"); return 2; }

    llama_batch batch = llama_batch_init(2048, 0, 1);
    auto decode_tokens = [&](const std::vector<llama_token> & toks, int pos_base, bool want_logits_last) -> bool {
        // decode in 2048-token chunks; logits only on the final token if asked
        size_t off = 0;
        while (off < toks.size()) {
            size_t chunk = toks.size() - off > 2048 ? 2048 : toks.size() - off;
            batch.n_tokens = 0;
            for (size_t i = 0; i < chunk; i++) {
                batch.token[batch.n_tokens] = toks[off + i];
                batch.pos[batch.n_tokens] = (llama_pos)(pos_base + off + i);
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] =
                    (want_logits_last && off + i + 1 == toks.size()) ? 1 : 0;
                batch.n_tokens++;
            }
            if (llama_decode(ctx, batch) != 0) return false;
            off += chunk;
        }
        return true;
    };

    // --- cached trajectory: prefill all but the last prompt token (no
    // logits), then decode the last prompt token with logits, then 1-token
    // greedy steps. This is the standard incremental (KV-cached) decode. ---
    std::vector<float> logits((size_t) n_gen * n_vocab);
    std::vector<int32_t> tokens(n_gen), tokens_nocache(n_gen);
    std::vector<llama_token> pre(prompt_tokens.begin(), prompt_tokens.end() - 1);
    if (!decode_tokens(pre, 0, false)) { fprintf(stderr, "prefill failed\n"); return 1; }
    std::vector<llama_token> gen;
    std::vector<llama_token> first = {prompt_tokens.back()};
    if (!decode_tokens(first, n_prompt - 1, true)) { fprintf(stderr, "first decode failed\n"); return 1; }
    for (int s = 0; s < n_gen; s++) {
        const float * lg = (s == 0) ? llama_get_logits(ctx) : nullptr;
        if (s > 0) {
            llama_token in = forced.empty() ? gen.back() : forced[s - 1];
            std::vector<llama_token> one = {in};
            if (!decode_tokens(one, n_prompt + s - 1, true)) {
                fprintf(stderr, "decode failed at step %d\n", s); return 1;
            }
            lg = llama_get_logits(ctx);
        }
        if (!lg) { fprintf(stderr, "no logits at step %d\n", s); return 1; }
        memcpy(&logits[(size_t) s * n_vocab], lg, (size_t) n_vocab * sizeof(float));
        int t = argmax_f32(lg, n_vocab);
        tokens[s] = t;
        gen.push_back((llama_token) t);
    }

    // --- no-cache trajectory: full recompute from scratch at every step ------
    std::vector<llama_token> prefix = prompt_tokens;
    for (int s = 0; s < n_gen; s++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode_tokens(prefix, 0, true)) { fprintf(stderr, "nocache decode failed at %d\n", s); return 1; }
        const float * lg = llama_get_logits(ctx);
        int t = argmax_f32(lg, n_vocab);
        tokens_nocache[s] = t;
        prefix.push_back(forced.empty() ? (llama_token) t : forced[s]);
    }

    char shape_logits[64], shape_toks[64];
    snprintf(shape_logits, sizeof(shape_logits), "(%d, %d)", n_gen, n_vocab);
    snprintf(shape_toks, sizeof(shape_toks), "(%d,)", n_gen);
    std::vector<ZipEntry> entries = {
        {"logits.npy", npy_buffer("<f4", shape_logits, logits.data(), logits.size() * sizeof(float))},
        {"tokens.npy", npy_buffer("<i4", shape_toks, tokens.data(), tokens.size() * sizeof(int32_t))},
        {"tokens_nocache.npy", npy_buffer("<i4", shape_toks, tokens_nocache.data(), tokens_nocache.size() * sizeof(int32_t))},
    };
    if (!write_npz(out_path, entries)) { fprintf(stderr, "failed to write %s\n", out_path.c_str()); return 1; }
    fprintf(stderr, "wrote %s (n=%d vocab=%d)\n", out_path.c_str(), n_gen, n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
