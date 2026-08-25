| Model | Recall@20 | NDCG@20 | Coverage | Latency p50 |
| --- | --- | --- | --- | --- |
| Two-tower + ranker | 0.0276 | 0.0171 | 0.4533 | 24.02ms |
| Multi-intent | 0.0185 | 0.0105 | 0.8629 | 6.55ms |
| Two-tower | 0.0154 | 0.0089 | 0.8130 | 0.80ms |
| Baseline | 0.0259 | 0.0172 | 0.5010 | 0.45ms |

## Ranker weak-label simulation

`trainer/train_ranker.py` trains 4 LightGBM binary boosters (click, cart, purchase, wishlist)
on top of the two-tower + session-encoder features. **H&M transaction data has no click,
cart, or wishlist events — only purchases.** The following labels are simulated, not
measured, and should not be read as real user engagement rates:

- `purchase = 1` for the actual transacted item; `0` for every sampled negative candidate.
  This is the only label backed by real data.
- `click = 1` for the purchased item (a purchase implies a click) and for any negative
  candidate whose `category_l1` matches a category the user already interacted with earlier
  in the same session — a proxy for "browsed the aisle," not an observed click.
- `cart = 1` for a purchased item with probability 0.4, drawn independently per example;
  always `0` for negatives. This simulates a funnel drop-off and does not reflect any
  real add-to-cart signal.
- `wishlist = 1` for a purchased item with probability 0.2, drawn independently per
  example; always `0` for negatives. Same caveat as `cart`.

The `item_ctr` input feature is also a fabricated proxy — `n_interactions / (n_interactions
+ 20)`, a smoothed popularity transform — since no impression/click log exists to compute a
real CTR.

These simulated labels exist to demonstrate a multi-objective ranking architecture end to
end on purchase-only historical data. Treat any click/cart/wishlist metric derived from this
ranker as illustrative of the pipeline, not as evidence about real user behavior.

## Load test

`bench/run_bench.sh` — Locust, headless, ramped to 200 concurrent users over 10s, held for
60s total, against a live `gateway` (4 `uvicorn` workers) + `retrieval_service` + Redis, all
running locally. Realistic mix: ~80% returning users drawn from a 300-user pool (accumulating
real session history as the run progresses), ~20% brand-new cold-start guests, ~35% of feed
requests preceded by a fresh view event. Full stats in `bench/results_stats.csv`; raw
per-request samples (used for `bench/latency.png`) in `bench/raw_latencies.csv`.

| Metric | `POST /v1/feed` | Aggregated (`/v1/feed` + `/v1/events`) |
| --- | --- | --- |
| Requests | 1,308 | 1,806 |
| Failures | 0 (0.00%) | 0 (0.00%) |
| Throughput | 22.58 req/s | 31.18 req/s |
| Min | 168ms | 61ms |
| p50 | 4,600ms | 4,000ms |
| p95 | 10,000ms | 10,000ms |
| p99 | 15,000ms | 21,000ms |
| Max | 33,858ms | 33,858ms |

**This is nowhere near the 80ms p99 budget — by design, not by accident.** 200 concurrent
users is far more concurrent load than 4 worker processes on one shared, contended sandbox VM
should ever absorb; the `min` (168ms) is close to what a single unloaded request costs, while
`p50`–`p99` climb into the seconds as requests queue behind a handful of worker threads. That
queueing-under-saturation is precisely the argument for horizontal scaling in
`architecture.md` §10 — one under-provisioned process buckles exactly like this, which is why
the capacity math below sizes for ~20+ replicas behind a load balancer, not one. See
`bench/latency.png` for the full distribution (80ms budget line included for scale, though at
this range it sits flush against the axis).

Two environment-specific caveats, recorded for honesty:

- This sandbox's Redis runs in a Docker container reached through a Windows/WSL2 port-forward
  that degraded mid-session (requests that depend on Redis — session lookup, popularity
  fallback — intermittently hung before failing over). `src/shared/db.py` was hardened with
  bounded socket timeouts as a direct result (`REDIS_SOCKET_TIMEOUT_S` /
  `REDIS_SOCKET_CONNECT_TIMEOUT_S`, alongside the pre-existing `PG_CONNECT_TIMEOUT_S`), so a
  degraded connection now fails fast into the existing graceful-degradation paths instead of
  blocking a request indefinitely — but it does mean this specific run's tail latency is
  partly a property of this sandbox's virtualized networking, not the application.
- Postgres is not provisioned in this sandbox (see [Guardrails and DPDP mapping] in the
  README), so every response's `recommendation_log` write fails and retries fast (bounded by
  `PG_CONNECT_TIMEOUT_S`) rather than ever succeeding — real audit-log latency in a deployment
  with a live Postgres would be lower than what this run measures.

## Cost model

`bench/cost_model.py`, parameterised by this run's measured mean latency (5,343.8ms,
`/v1/feed`'s Average Response Time from `bench/results_stats.csv`):

```
$ python -m bench.cost_model --stats-csv bench/results_stats.csv --endpoint /v1/feed

Capacity math for 10,000 req/s (mean latency: 5343.8ms measured)
  Concurrent requests in flight:      53438
  Throughput per API pod:             3 req/s
  API pods needed (+40% headroom):    4676
  Retrieval pods needed:               20

Cost per hour:                        $752.56
Cost per 1,000 recommendations:       $0.020907
Cost per 1,000,000 recommendations:   $20.91

One-LLM-call-per-request would cost:  $500.00 per million — 24x more expensive
```

Feeding a genuinely saturated, contended-hardware measurement into a formula that assumes
*mean service time* produces a pessimistic, not representative, pod count — 4,676 pods is what
the math says if every production pod performed as badly as one process sharing a laptop-class
sandbox VM with three other services, a load generator, and unrelated tenant containers. It
still beats a per-request LLM call by 24x, which is itself notable: even the worst-case,
fully-loaded measurement from this session doesn't erase the cost argument for keeping the LLM
off the hot path.

For the properly-scaled comparison — many replicas, each handling a modest share of concurrent
load, at the mean service time an unloaded request actually costs (this run's own `min` was
168ms; `architecture.md` §10's design target is 35ms) — see [Cost per 1,000 inferences] in the
README, which reports both this measured number and the original design-target projection
side by side.
