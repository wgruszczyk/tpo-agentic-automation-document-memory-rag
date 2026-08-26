---
name: product-memory-operations
description: 'Use when running, feeding, diagnosing or measuring this Product Memory RAG service: adding knowledge folders, files not appearing in the index, ingestion failures or skipped documents, running an evaluation, generating a question set, comparing embedding models, reading Grafana or MLflow, or changing a model safely.'
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

Grafana on 2601 for query latency by stage, index growth and logs; MLflow on 2604 for evaluation
runs and their comparison. `LOG_FORMAT=json` is required for Loki to index log fields.

`histogram_quantile` over a rate returns nothing until Prometheus has scraped a *changing* counter
twice, so panels look broken for the first minute after a restart.

Full reference: [README](../../../README.md).
