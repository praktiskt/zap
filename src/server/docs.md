# zap — usage

zap is an in-memory full-text search engine (BM25). One binary, no
dependencies: documents live in RAM, served over HTTP. This page is served
live by the node itself at `GET /docs`.

## quickstart

```bash
zap serve --bind 0.0.0.0:3279 &
curl -X POST localhost:3279/ingest \
  -d '{"text": "the quick brown fox jumps"}'
curl 'localhost:3279/search?q=fox&k=5'
```

## ingest — `POST /ingest`

Accepts NDJSON (one object per line) or a JSON array. Commits
automatically, so documents are searchable in the reply. Response:
`{"accepted": N, "docs": TOTAL, "committed": true}`.

```bash
# ndjson
curl -X POST localhost:3279/ingest --data-binary @docs.ndjson
# array
curl -X POST localhost:3279/ingest -d '[{"text": "..."}, {"text": "..."}]'
```

Document shape — `text` is required, `id` and `metadata` optional:

```json
{"text": "the quick brown fox", "id": "doc-1", "metadata": {"lang": "en"}}
```

Re-ingesting an existing `id` **replaces** the document. Unknown fields
are rejected with 400.

## search — `GET /search?q=...&k=10` or `POST /search`

```bash
curl 'localhost:3279/search?q=quick%20brown&k=5'
curl -X POST localhost:3279/search -d '{"q": "quick brown", "k": 5}'
```

Response: `{"results": [{"score":.., "id":.., "docid":.., "doc":..,
"metadata":..}], "count": N}` — `doc` is the stored text, `id`/`metadata`
only when the document carries them.

### query syntax

| form | meaning |
|---|---|
| `fox` | term (stemmed) |
| `"quick brown"` | phrase, exact order |
| `"quick brown"~2` | phrase with slop 2 |
| `fox AND dog` / `fox dog` | both (adjacency = AND) |
| `fox OR dog` | either |
| `NOT fox` / `fox AND NOT dog` | exclusion |
| `(fox OR dog) AND lazy` | grouping |
| `qui*` | prefix |
| `f?x`, `br*n` | wildcard (`?` = 1 char) |
| `fox~` / `fox~2` | fuzzy (auto distance 1, or explicit) |
| `title:fox` | restrict to one field |

## documents — `GET /doc/{id}`, `DELETE /doc/{id}`

Fetch or delete by external id (cluster-aware: routed to the owning
shard). Deleting a large share of the corpus triggers a purge merge
automatically; small deletes stay masked until the segment-count trigger.

## ops — `/stats`, `/commit`, `/health`, `/dashboard`

- `GET /stats` → `docs`, `segments`, `generation`, `searches`,
  `ingested`, `uptime_s` (+ per-shard rows in cluster mode).
- `POST /commit` — seal ingested documents into searchable segments
  (`/ingest` already commits per request; explicit commits matter for
  batched/library use).
- `GET /health` → `{"status":"ok"}` (liveness/readiness probes).
- `GET /dashboard` — live ops page (stats, ingest/search forms).
- `GET /docs` — this page (`text/markdown`).

## cluster mode

```bash
zap serve --bind 0.0.0.0:3279 --peers <svc>:3279 \
  --announce <pod-ip>:3279 --resync
```

Any pod fans searches out (`/__zap/shard_search`) and ring-routes
ingests/reads by id hash, so every node serves the global view. BM25
stats gossip at commit; `--resync` pulls the node's ring share on boot;
shard moves via `POST /__zap/peers`. `GET /__zap/handoff` streams a
shard's live docs for moves/recovery. RAM-only: shards hold no copy of
each other — this moves data, it does not replicate it. Add `--disk DIR`
for reboot-safe segments (manifest + content-addressed files).

`/__zap/*` endpoints are cluster internals (never forwarded, always
answered by the local shard).

## CLI

```bash
zap --help                 # global flags: --lanes, --top-k, --k1, --b,
                           #   --no-stem, --disk DIR, --commit-every
zap corpus/                # index files, then search interactively
zap serve [inputs...]      # index inputs, then serve
```

Interactive commands: `:help`, `:stats`, `:quit`. Snippet width, plain
output and stemming are CLI flags, not query parameters.
