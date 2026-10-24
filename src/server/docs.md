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

Extra keys: a **string** is another text field, an **array of numbers** is an
embedding (see "embedding search" below).

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

## embedding search

zap stores and searches the embeddings **you** compute — the model stays on
your side. Declare the vector field when the node boots:

```bash
zap serve --bind 0.0.0.0:3279 --vector emb:768          # cosine (default)
zap serve --bind 0.0.0.0:3279 --vector emb:768:dot      # or l2
```

Ingest: an array of numbers is the embedding (the dimension must match the
declared one — otherwise 400).

```bash
curl -X POST localhost:3279/ingest \
  -d '{"id":"doc-1","text":"the quick brown fox","emb":[0.012,-0.44, ...]}'
```

Search: send the query embedding.

```bash
# embedding only
curl -X POST localhost:3279/search -d '{"vector":[0.012,-0.44, ...],"k":5}'
# keyword AND embedding, fused
curl -X POST localhost:3279/search \
  -d '{"q":"quick brown","vector":[0.012,-0.44, ...],"mode":"hybrid","alpha":0.5}'
# the keyword selects the candidates, the embedding ranks them
curl -X POST localhost:3279/search \
  -d '{"q":"quick brown","vector":[0.012,-0.44, ...],"mode":"filter"}'
```

| parameter | meaning |
|---|---|
| `vector` | the query embedding (array of numbers; on GET: comma separated) |
| `field` | which vector field to search (default: the first one) |
| `mode` | `keyword` \| `vector` \| `hybrid` \| `filter` (default: hybrid when both `q` and `vector` are given) |
| `alpha` | keyword weight in the weighted fusion (0 = embedding only, 1 = keyword only) |
| `fusion` | `weighted` (default) or `rrf` (reciprocal rank fusion) |

Result rows carry the components: `score` (the fused score), `keyword_score`
and `vector_score` (null when that side did not produce the document).

Note on `weighted`: each side is min-max normalized over the candidate
pool, so with few matches a small keyword difference is amplified to the
full range. `rrf` is scale-free and does not have that property.

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
                           #   --no-stem, --disk DIR, --commit-every,
                           #   --vector NAME:DIM[:METRIC]
zap corpus/                # index files, then search interactively
zap serve [inputs...]      # index inputs, then serve
```

Interactive commands: `:help`, `:stats`, `:quit`, and for the embedding
side `:vec <csv>` (set the query embedding), `:vec off`, `:mode
keyword|vector|hybrid|filter`, `:alpha <0..1>`. Snippet width, plain
output and stemming are CLI flags, not query parameters.
