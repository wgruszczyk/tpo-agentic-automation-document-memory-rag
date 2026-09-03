---
name: product-memory-operations
description: 'Use when running, feeding, diagnosing or measuring this Product Memory RAG service: adding knowledge folders, files not appearing in the index, ingestion failures or skipped documents, running an evaluation, generating a question set, comparing embedding models, reading Grafana or MLflow, changing a model safely, or setting up and troubleshooting the local chat model and chat UI.'
argument-hint: 'What do you want to do — add knowledge, diagnose ingestion, or measure quality?'
---

# Product Memory Operations

Operating the local RAG service in this repository. All commands run from the repository root and
assume `make start` has been run at least once.

## Adding knowledge

Files placed in `knowledge/` are picked up within `SCAN_INTERVAL_SECONDS`. To index a folder that
lives elsewhere, do **not** create symlinks by hand: a bare symlink resolves on the host but dangles
inside the container, and nothing gets indexed.

```dotenv
KNOWLEDGE_LINKED_DIRS=Name=/absolute/path;Other=~/another/path
```

```bash
make link-knowledge
docker compose up -d --force-recreate product-memory
```

`make link-knowledge` creates the symlinks *and* regenerates `docker-compose.override.yml` with a
read-only bind mount for each target, which is what makes them readable in the container. The
force-recreate is required: a plain restart does not pick up new mounts.

Link names become the first segment of every `source_path`, which is the document identity and what
evaluation questions match on. **Renaming a link re-indexes its documents as new ones and invalidates
any question set referring to them.**

## When documents do not appear

```bash
make skipped    # readable, but held no indexable text
make failures   # could not be read at all, with the error
```

Both are recorded lists, reported in the log only when they change. Common causes:

- **`OSError: [Errno 35] Resource deadlock avoided`** — a cloud placeholder. OneDrive and iCloud keep
  files "online only" and a Docker bind mount cannot trigger the download. Set the folder to *Always
  Keep on This Device*; the next scan picks the files up with no reindex.
- **Files starting with `~$`** are never read. An office suite writes those beside an open document;
  they carry a real extension and hold no document.
- Pictures with no readable text, empty files and password-protected workbooks are skipped, which is
  normal rather than a fault.

## Measuring quality

Never judge a ranking or model change by inspection. Score it.

```bash
make generate-eval args="--seed <new-seed> --count 60"   # build a question set
cp eval/questions.generated.yaml eval/questions.yaml
make eval args="--track --run-name what-changed"
```

Three things that will mislead you if forgotten:

1. **Changing the corpus invalidates the baseline.** Runs are tagged with the index fingerprint;
   only compare runs sharing one.
2. **Regenerating without changing `--seed` re-picks the same documents.** Sampling is deterministic
   per seed, so a "fresh" set after adding documents is the old set unless the seed changes.
3. **Generated questions cannot judge anything that changes how much text a stage reads** —
   `RERANKER_MAX_LENGTH`, `CHUNK_SIZE`. Each probe is drawn from a sentence anywhere in a chunk, so
   they reward reading further into a chunk for its own sake. Keep a handwritten set and check any
   result against it; the two have moved in opposite directions on exactly this.

Read metrics as: `hit_rate` did an expected document come back at all, `mrr` how near the top,
`recall` how many of them, `precision` falls as `top_k` rises, `ndcg` needs grades to mean anything.
Below roughly 50 questions, a few points is indistinguishable from luck.

`make eval args="--answers"` scores the written answers on the same set: `citation_coverage` and
`citation_precision` are arithmetic over the sources each answer used. Adding `--judge <model>` asks
a local model whether each answer stayed inside those sources. That number is a trend, not a fact —
a judge from the same family as the writer flatters it, so use the largest model that fits and never
the identical one being scored. Answers take minutes where retrievals take seconds.

## Conversation

Retrieval is unaffected by any of this; conversation is a layer on top and off by default.

Ollama runs **on the host**, never in the container: Docker on macOS cannot reach the GPU, so a
containerised model falls back to the CPU and is unusable.

```bash
OLLAMA_HOST=0.0.0.0 ollama serve   # 0.0.0.0, or the container cannot reach it
ollama pull qwen3:8b
make chat-check                    # is it up, and does it hold what .env asks for?
make chat                          # Open WebUI on 2605
```

Set `CHAT_ENABLED=true` in `.env` and `make restart` first; `make chat` refuses otherwise.

Things that will waste your time if forgotten:

- **Connection refused from inside the container** — Ollama defaults to listening on `127.0.0.1`,
  which is not reachable across the container boundary. `OLLAMA_HOST=0.0.0.0`.
- **A non-local `OLLAMA_BASE_URL` fails at start-up, deliberately.** Only loopback, the container
  host, and private addresses are accepted. This is the guarantee that documents stay on the machine.
- **16 GB is tight.** The container already holds the embedding model, the reranker and Postgres.
  If answers crawl, the machine is swapping: drop to `qwen3:4b`, shorten `CHAT_KEEP_ALIVE`, or leave
  `CHAT_CONDENSE_MODEL` empty so no second model stays resident.
- **"I have nothing in the knowledge base"** is `CHAT_REQUIRE_EVIDENCE` working, not a bug. Nothing
  cleared `MIN_SEMANTIC_SCORE`. Check with `make query q='...'` before blaming the model.
- **Follow-ups that retrieve nothing** mean the condensing stitch was not enough. Set
  `CHAT_CONDENSE_MODEL=qwen3:1.7b` for a real rewrite.
- **Open WebUI's own RAG and its Ollama access are switched off on purpose.** Re-enabling either
  gives you a model that looks identical in the UI and answers from its training instead of your
  documents.

## Changing a model

A full reindex re-embeds every chunk and holds the service down while it runs, so measure before
paying for it:

```bash
make compare-embeddings model=<candidate>
```

This scores the candidate against the current model on cosine alone, without re-embedding the index.
A candidate that separates no better here will not repay a reindex. Bigger is not reliably better —
larger embedding and reranker models have both measured worse on real corpora here.

To adopt one, `make warmup` downloads it and prints the revision to pin in `.env`. That is the only
command that reaches the internet when `ALLOW_MODEL_DOWNLOAD=false`.

## Watching it run

```bash
make observability      # Prometheus, Loki, Promtail, Grafana, MLflow
make metrics            # raw scrape, when a dashboard panel is empty
```

Grafana on 2601 for query latency by stage, index growth and logs; its **Conversation** row covers
time to first token, where an answer's time goes, and outcomes — `no_evidence` there is the index
refusing to guess. MLflow on 2604 for evaluation runs and their comparison. `LOG_FORMAT=json` is
required for Loki to index log fields.

`histogram_quantile` over a rate returns nothing until Prometheus has scraped a *changing* counter
twice, so panels look broken for the first minute after a restart.

Full reference: [README](../../../README.md).
