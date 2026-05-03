# llm-speed Privacy

Every byte the CLI uploads is listed below. Preview the exact JSON for any
run before it ships:

```
llm-speed bench --quick --dry-run --print-payload
```

---

## 1. What we collect by default

Each upload contains exactly these fields. No others.

### Hardware fingerprint (bucketed / coarse)

| Field | Example | Why |
|---|---|---|
| `os_name` | `Darwin`, `Linux`, `Windows` | leaderboards filter by OS family |
| `os_version` | `26`, `13`, `6` (major version only) | major-version-level perf differences are real; patch level is identifying |
| `cpu_model` | `Apple M3 Pro`, `AMD Ryzen 9 7950X3D` | the canonical CPU label users compare on |
| `cpu_cores` | `12` | CPU-bound workloads scale here |
| `ram_gb` | `40` (rounded to nearest 8 GB) | memory headroom matters; serial-number-grade precision doesn't |
| `gpus[].name` | `RTX 4090`, `M3 Pro`, `MI300X` | the canonical GPU label |
| `gpus[].kind` | `nvidia`, `amd`, `apple`, `intel`, `cpu` | tells the leaderboard which family |
| `gpus[].memory_gb` | `24` (rounded to nearest 8 GB) | VRAM bucket determines what models fit |
| `accelerator_summary` | `M3 Pro (18-core GPU) + 36GB unified` | human-readable headline for the run page |
| `fingerprint_hash` | `a1b2c3d4e5f60708` (16 hex chars) | groups runs of the same hardware class for outlier detection. Hashed over only the bucketed fields above — two physically-different machines with the same SoC + RAM bucket + major OS produce the SAME hash. NOT user-level identity. **Omitted entirely in `--strict-anon` mode.** |
| `extras.backends` | `{"llama.cpp": "...", "metal": "Metal 3"}` | so we can correlate decode tps to backend version |

### Workload results (the actual numbers)

For each workload (W1..W5):

| Field | Why |
|---|---|
| `workload` | which W1..W5 |
| `suite_version` | which methodology version (`suite-v1`, ...) |
| `backend` / `backend_version` | llama.cpp / ollama / vllm / mlx / hosted-api + version |
| `model.{name,size,quant,digest}` | the (model x quant) tuple is the leaderboard axis |
| `ttft_ms`, `prefill_tps`, `decode_tps`, `decode_p50_latency_ms`, `decode_p95_latency_ms` | the benchmark metrics |
| `prompt_tokens`, `output_tokens`, `batch_size`, `context_tokens` | workload params for replay |
| `wall_ms` | end-to-end timing |
| `prefix_cache_hit_rate` | when backend exposes it |
| `error`, `flags` | captured failure mode + thermal/battery flags |
| `extras` | small backend-specific telemetry; capped at 256 chars per string field by the client-side privacy invariant (see below) |

### Provenance

| Field | Why |
|---|---|
| `cli_version` | so backend-version-specific anomalies are diagnosable |
| `started_at`, `finished_at` | run-duration sanity check |
| `signature` (ed25519) | proves the bytes weren't edited after signing |
| `public_key` (ed25519, base64 32B raw) | server verifies signature with this; persistent in default mode, ephemeral per run in `--strict-anon` |

---

## 2. What we don't collect

The CLI does not send any of the following. They never leave your machine:

- **PCI bus IDs** (`pci.bus_id` from `nvidia-smi`)
- **GPU driver build numbers** (full `driver_version` strings)
- **GPU vBIOS versions**
- **Kernel patch versions** (`6.8.0-31-generic` → just `6`)
- **OS patch level** (`26.3.1` → just `26`)
- **macOS marketing minor/patch** (`13.5.7` → just `13`)
- **Linux distro patchlevel** (`Ubuntu 24.04.1 LTS` is dropped entirely)
- **Locale**
- **NVMe / disk model / serial**
- **Hostname / username**
- **Power source / thermal state** (captured locally for run flags, never sent)
- **Prompt text** — the workload prompts are part of the open-source suite, not user data
- **Model output text** — never captured at all
- **Per-token raw timings** (`raw_timings_ms`) — kept locally for replay/dispute, only included in upload if you opt in via `include_raw_timings=True`

### Privacy invariant (client-side)

Before any upload, the CLI walks the payload tree and refuses to send if any
non-whitelisted string field exceeds 256 characters. The whitelist:
`accelerator_summary`, `error`, `fingerprint_hash`, `public_key`, `signature`,
`suite_version`, `cli_version`, `model.{identifier,name,size,quant,digest}`,
`workload`, `backend`, `backend_version`. Anything else aborts the upload with
`upload payload contains unexpected long string at <path>; refusing to upload (privacy invariant)`.

---

## 3. Three modes

| Capability | default | `--anon` | `--strict-anon` |
|---|---|---|---|
| Hardware bucket (OS major, CPU, RAM bucket, GPU name + VRAM bucket) | yes | yes | yes |
| `accelerator_summary` | yes | yes | yes |
| `fingerprint_hash` (class-level, bucketed) | yes | yes | **no** |
| `public_key` (persistent across your runs) | yes | yes | **no — ephemeral, rotated per run** |
| `User-Agent: llm-speed-cli/<version>` | sent | sent | **not sent (default httpx UA)** |
| `Authorization: Bearer <api-key>` | if provided | **not sent** | not sent |
| `X-LLM-Speed-Anon: 1` header | no | yes | no (would be a beacon) |
| Server-side IP logging | not persisted | not persisted | not persisted |
| Cross-run correlation by the server | possible via `public_key` and `fingerprint_hash` | possible via `public_key` | **not possible** |

### `--anon` vs `--strict-anon`

`--anon` is "soft anonymous": the persistent ed25519 keypair (and therefore
`public_key` / `fingerprint_hash`) still ships, so the server can group your
runs together. The `Authorization: Bearer <api-key>` header is dropped — your
contributor account is not linked to the upload. Use `--anon` if you don't
want the leaderboard to attribute the run to you but you're fine with the
cross-run-correlation property the persistent keypair provides.

`--strict-anon` is "no identity at all": ephemeral keypair per run, no
fingerprint hash, default User-Agent.

### `--strict-anon` in detail

- A fresh ed25519 keypair is generated **in memory** for each run. The persistent
  keypair on disk (`~/.config/llm-speed/keys/ed25519.key`) is never read or written.
- No `fingerprint_hash` in the payload. Two consecutive `--strict-anon` runs
  from the same machine are unlinkable from the server's perspective.
- The signature still proves the bytes weren't tampered with after signing —
  it just doesn't tie the run to a long-term identity.

### `--no-upload`

- No network call. The result is saved to
  `~/.cache/llm-speed/runs/<isoformat>-<fp_hash>.json`.
- Re-upload later with `llm-speed bench --resume <path>` (which verifies the
  signature locally before posting).

---

## 4. Server-side

The fields the server stores are documented at
[`/privacy.json`](https://api.llm-speed.com/privacy.json) (machine-readable,
versioned).

- **We don't persist client IPs.** Not in the database, not in application logs.
  Inbound request metadata (User-Agent, request path, response code) is logged
  at the Cloudflare-platform level per [Cloudflare's privacy policy](https://www.cloudflare.com/privacypolicy/).
  We don't export, query, or retain those edge logs.
- **Rate-limiting uses an ephemeral hashed-IP counter.** The hash is salted with
  a value that rotates daily and is held in memory only — never written to
  storage, never returned in any response. Cross-day correlation by IP is not
  possible.
- The server has no access to anything the CLI didn't put in the payload.
  Driver builds, PCI IDs, kernel patches, etc. are physically absent.

---

## 5. How to inspect what's sent

```
llm-speed bench --quick --dry-run --print-payload
```

`--dry-run` runs the workloads but skips both upload and local save.
`--print-payload` prints the exact JSON that would be uploaded to stdout.

You can also save without uploading and pretty-print the file:

```
llm-speed bench --quick --no-upload --json /tmp/run.json
cat /tmp/run.json | jq
```

The bytes signed by the CLI are exactly the bytes between the outer braces of
that JSON, modulo the `signature` and `public_key` fields (which are excluded
from canonical hashing). The server verifies against the same canonicalization.

---

## 6. How to delete your data

- **Self-serve consent revocation:** `rm ~/.config/llm-speed/consent.json`
  — the next upload will re-prompt.
- **Email** privacy@llm-speed.com with the `id` (the `r_xxxxxxxx` slug) of any
  result you want removed. We remove it within 7 days.
- **Public dispute thread:** every result page has a dispute link. File a
  challenge; if it stands, the result is withdrawn.

`--strict-anon` submissions can't be linked to your machine after the fact
(by design — there's no persistent identity to match against), so deletion of
those runs requires the run `id` from your local cache.

---

## 6a. GDPR / EU data protection

For visitors and submitters in the EU, EEA, and UK:

- **Data controller:** `meadow-kun`, the maintainer of llm-speed.com,
  reachable at privacy@llm-speed.com or via GitHub issues at
  [meadow-kun/llm-speed](https://github.com/meadow-kun/llm-speed/issues).
  This is a single-maintainer open-source project; there is no separate
  legal entity behind the domain.

- **Legal basis (Article 6):** legitimate interest under Art. 6(1)(f) for
  the benchmark payload (necessary to operate a public benchmark database
  whose value depends on aggregating submitted runs). First-run consent is
  also captured locally in `~/.config/llm-speed/consent.json` before any
  upload, so the same processing also has consent under Art. 6(1)(a).

- **What's processed and retained:**
  - **Benchmark payloads** (the fields enumerated in §1): retained
    indefinitely so historical leaderboards remain meaningful. You can
    request deletion of any specific run at any time by emailing the
    address above with the run id.
  - **Edge access logs** (Cloudflare's platform-level logs covering IP +
    User-Agent + path + response code): governed by
    [Cloudflare's privacy policy](https://www.cloudflare.com/privacypolicy/);
    we do not export, query, or retain those logs ourselves.
  - **Rate-limit hashed-IP counter:** in-memory only, dropped after one
    minute; daily salt rotated and never persisted (see §4).
  - **Site analytics** (Cloudflare Web Analytics, see §8): aggregated
    pageview counts, no per-visitor identifiers, no cookies.

- **Your rights under the GDPR:**
  - Access (Art. 15) — request a copy of any benchmark payload tied to a
    public key you control.
  - Rectification (Art. 16) — file a public dispute on the run page.
  - Erasure (Art. 17) — request deletion via privacy@llm-speed.com with
    the run id.
  - Portability (Art. 20) — every benchmark payload is already public
    JSON at `https://api.llm-speed.com/v1/results/<id>` and signed by your
    own keypair.
  - Right to lodge a complaint with a supervisory authority (Art. 77).

- **International transfers:** benchmark payloads live on Cloudflare D1
  (SQLite-on-edge), replicated globally across Cloudflare's network for
  read performance. No primary copy in any specific jurisdiction; serving
  is from the closest edge.

- **No automated decision-making.** Submitted runs are aggregated and
  ranked; no individual visitor is profiled or scored.

- **Children:** this service is not directed at children under 16. We do
  not knowingly process data from children.

If your submission was made on a work machine and you need a sentence
your DPO can point at: "The processing has a legitimate-interest legal
basis (Art. 6(1)(f) GDPR), the data is technical-benchmark-only with no
personal identifiers in the payload (see §1 and §2), retention is
indefinite for the run record but immediate-deletion-on-request is
supported, and the controller can be reached at privacy@llm-speed.com."

---

## 7. First-run consent

The first time you run `llm-speed bench` against the public API in a mode that
will upload (i.e. not `--no-upload`, not `--dry-run`, not `--json`, not
`--strict-anon` — strict-anon is implicit consent because it's the most-private
option), the CLI prints the consent text and asks `[Y/n]`. The choice is saved
to `~/.config/llm-speed/consent.json` (mode 0600) along with a timestamp, the
CLI version, and the trimmed `fingerprint_hash`. Delete that file to revoke.

---

## 8. Site analytics

The website (llm-speed.com) uses **Cloudflare Web Analytics** to count
pageviews. It is the only analytics product running on the site.

| Aspect | Value |
|---|---|
| Provider | Cloudflare, Inc. (the same edge that already terminates TLS for the site) |
| Cookies | none |
| `localStorage` / `sessionStorage` | none |
| Client-side fingerprinting | none |
| Persistent client identifier | none |
| Client IP storage | not persisted (Cloudflare aggregates at the edge; the unaltered edge-log retention applies as already documented in §4) |
| Beacon | edge-injected by Cloudflare Pages; no `<script>` runs in your browser to count a pageview |
| Data fields collected | request path, response code, referring host (not full URL), country, browser family, page-load timing |
| Data NOT collected | search query terms, full referrer URL, screen resolution, mouse movements, session replay, cross-site identity |
| Retention | Cloudflare aggregates, ~6 months on the free tier; `docs/metrics/<iso>-analytics.json` snapshots in this repo retain 7-day aggregates indefinitely as a public log |
| API token scope | `Account Analytics: Read`, account-scoped to llm-speed only, used by `tools/metrics_analytics.py` for the daily snapshot |
| Opt-out | block `static.cloudflareinsights.com` (the optional RUM beacon endpoint, only active if the maintainer enables web-vitals tracking — not active today). The edge counter cannot be opted out of without blocking the entire site, but it never sees identifiable data. |

If we ever turn on the optional RUM beacon (web vitals — LCP / CLS / INP),
this document and the CSP get updated in the same commit. We don't ship
analytics features without disclosing them here.

---

## 9. Open source

Every line of code that handles your data is in the public repo. Auditable
entry points:

- Fingerprint capture and bucketing: [`cli/fingerprint.py`](https://github.com/meadow-kun/llm-speed/blob/main/cli/fingerprint.py)
- Upload payload construction and signing: [`cli/upload.py`](https://github.com/meadow-kun/llm-speed/blob/main/cli/upload.py), [`cli/signing.py`](https://github.com/meadow-kun/llm-speed/blob/main/cli/signing.py)
- Server-side ingest and signature verification is closed-source.
  The CLI's JWS payload (Ed25519 over canonicalised JSON) is verifiable
  against the public key in the JWS protected header by any
  RFC-7515-compliant library — you don't need our server code to confirm
  a signature.

File an issue if anything in this document doesn't match the code.
