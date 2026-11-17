"""Seeded, deterministic generator of a fictional operations world.

``generate_world(seed)`` returns a :class:`World`: ~33 text "files" (service
registry, configs, logs, deploy logs, incident tickets, on-call roster, a DB
dump, vendor invoices, runbooks, postmortems, a change calendar, a people
directory, an architecture note) plus ``facts``, the structured ground truth
that question generators join over.

Design goals:

* Determinism: only ``random.Random(seed)`` is used, and no set iteration
  order leaks into the output.
* Retention pressure: every file is ~1.4k-4.2k chars (≈400-1200 tokens); the
  corpus is ≈28-32k tokens, so reading everything forces several compactions
  at a 16k context cap.
* Distractors: near-duplicate service names (``ledger-api`` / ``ledger-worker``)
  with digit-swapped ports, superseded and staging config values, warning
  thresholds next to action thresholds, similar company names, people sharing
  a first name, a "former staff" table, rollbacks in the deploy log.
* Every planted identifier is drawn through ``_Builder.uniq`` so it is unique
  across the whole world (no accidental collisions that would make a question
  ambiguous).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

__all__ = ["World", "estimate_tokens_text", "generate_world"]


def estimate_tokens_text(text: str) -> int:
    """Same heuristic as ``compaction.tokens``: ~3.5 chars per token."""
    return -(-len(text) * 2 // 7)


@dataclass
class World:
    seed: int
    company: str
    files: dict[str, str]  # path -> content, in generation order
    reading_order: list[str]  # recommended reading order (the checklist)
    facts: dict = field(default_factory=dict)

    def corpus_text(self) -> str:
        return "\n".join(self.files[p] for p in self.reading_order)

    def corpus_chars(self) -> int:
        return sum(len(t) for t in self.files.values())

    def corpus_tokens(self) -> int:
        return sum(estimate_tokens_text(t) for t in self.files.values())


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------

COMPANIES = [
    ("Northwind Freight", "northwind"),
    ("Kestrel Logistics", "kestrel"),
    ("Bluefin Couriers", "bluefin"),
    ("Arclight Shipping", "arclight"),
    ("Tamarack Haulage", "tamarack"),
    ("Saltmarsh Express", "saltmarsh"),
]

FIRST = [
    "Amara", "Bjorn", "Chidi", "Dana", "Elif", "Farid", "Greta", "Hiro",
    "Ines", "Jonas", "Kemi", "Lars", "Mei", "Nadia", "Oskar", "Priya",
    "Quinn", "Rafael", "Sana", "Tomas", "Uma", "Viktor", "Wen", "Ximena",
    "Yusuf", "Zofia",
]
LAST = [
    "Okafor", "Lindqvist", "Nakamura", "Petrov", "Haddad", "Moreau",
    "Castillo", "Brennan", "Varga", "Ilunga", "Sorensen", "Achebe",
    "Kowalski", "Ferreira", "Tanaka", "Mbeki", "Novak", "Albrecht", "Desai",
    "Rinaldi", "Halloran", "Qureshi",
]
TEAMS = ["platform", "payments", "logistics", "data", "sre"]
TITLES = ["Software Engineer", "Senior Software Engineer", "Staff Engineer",
          "Site Reliability Engineer", "Engineering Manager", "Data Engineer"]
LOCATIONS = ["Rotterdam", "Lisbon", "Toronto", "Nairobi", "Remote (EU)", "Remote (US)"]

SERVICE_PAIRS = [
    ("ledger-api", "ledger-worker"),
    ("route-planner", "route-planner-v2"),
    ("auth-gateway", "auth-proxy"),
    ("billing-sync", "billing-export"),
    ("notify-hub", "notify-relay"),
    ("inventory-svc", "inventory-sync"),
    ("search-indexer", "search-api"),
    ("media-resizer", "media-cdn-edge"),
    ("fleet-tracker", "fleet-telemetry"),
]

VENDORS = [
    "Corvid Cloud Hosting", "Lumen Fibre Ltd", "Paperhill Office Supply",
    "Brightlane Security", "Quarry Data Labs", "Helix Telecom",
    "Summit HVAC Services", "Orbital Pager Co",
]

ACC_ADJ = ["Granite", "Silver", "Copper", "Harbor", "Juniper", "Maple", "Cobalt",
           "Willow", "Falcon", "Amber", "Cedar", "Orchid"]
ACC_MID = ["Peak", "Park", "Ridge", "Bay", "Field", "Point", "Creek", "Vale"]
ACC_SUF = ["Outfitters", "Grocers", "Pharma", "Textiles", "Furniture", "Brewing",
           "Electronics", "Garden Supply"]

INCIDENT_SYMPTOMS = [
    "Elevated 502s", "Connection refusals", "Latency spike", "Request timeouts",
    "Partial outage", "Error-rate burst",
]
HEX = "0123456789abcdef"

PROSE = [
    "Dashboards for this area live in the shared Grafana folder; ask in #ops-help if a panel looks stale.",
    "Staging traffic is synthetic and must not be used to judge production latency.",
    "All times in this document are UTC unless stated otherwise.",
    "If anything here contradicts the service registry, the registry wins and this page should be fixed.",
    "Please keep edits small and link the relevant ticket in the change description.",
    "The quarterly review flagged this area as needing better alert hygiene.",
    "Most of the noise in this channel comes from flapping health checks during deploys.",
    "We agreed to revisit the retry budget once the new load balancer pool is fully rolled out.",
    "Synthetic probes run every minute from three regions and page only after two consecutive failures.",
    "Customer support should be told before any planned maintenance longer than fifteen minutes.",
    "Feature flags are evaluated at request time, so toggling one does not require a restart.",
    "The legacy cron host is scheduled for decommissioning and should not receive new jobs.",
    "Log retention for production is thirty days in the hot tier and one year in cold storage.",
    "When in doubt, roll back first and investigate second.",
    "The read replicas lag slightly behind the primary during the nightly batch window.",
    "Capacity planning numbers are refreshed at the start of every month.",
    "Secrets are injected by the platform sidecar; never commit them to the repository.",
    "Load tests must be announced in #perf at least one day in advance.",
    "Runbook owners review their pages after every incident that touched them.",
    "Traffic from the mobile apps peaks around the evening delivery window.",
    "Alert thresholds were tuned after last winter's peak season and have not changed since.",
    "The vendor status page is often slower to update than our own probes.",
    "Old dashboards were archived during the observability migration; links may 404.",
    "Changes to shared libraries need a review from the platform team.",
    "Escalate to the engineering manager on duty if the primary does not acknowledge within ten minutes.",
]


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def _fmt_ts(dt: datetime, ms: bool = False) -> str:
    if ms:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class _Builder:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.used: set[str] = set()

    # -- helpers -----------------------------------------------------------
    def uniq(self, fn):
        for _ in range(10_000):
            v = fn()
            if str(v) not in self.used:
                self.used.add(str(v))
                return v
        raise RuntimeError("could not draw a unique value")

    def hexs(self, n: int) -> str:
        return "".join(self.rng.choice(HEX) for _ in range(n))

    def prose(self, k: int) -> str:
        return " ".join(self.rng.sample(PROSE, k))

    def pad(self, head: str, filler, target: int, tail: str = "") -> str:
        """Append ``filler()`` lines after ``head`` until ~target chars."""
        out = head
        budget = target - len(head) - len(tail)
        guard = 0
        while guard < 400:
            guard += 1
            line = filler()
            if len(line) + 1 > budget:
                if budget < 60:
                    break
                continue
            out += line + "\n"
            budget -= len(line) + 1
        return out + tail

    def target(self, lo: int = 2300, hi: int = 4050) -> int:
        return self.rng.randint(lo, hi)

    def rand_dt(self, start: datetime, end: datetime) -> datetime:
        span = int((end - start).total_seconds())
        return start + timedelta(seconds=self.rng.randint(0, span))

    # -- world ---------------------------------------------------------------
    def build(self) -> World:
        rng = self.rng
        self.company, self.slug = rng.choice(COMPANIES)
        year = rng.choice([2025, 2026])
        month = rng.randint(2, 9)
        d0 = date(year, month, 1)
        while d0.weekday() != 0:  # first Monday
            d0 += timedelta(days=1)
        self.window_start = datetime(d0.year, d0.month, d0.day, 9, 0, 0)
        self.n_weeks = 8
        self.window_end = self.window_start + timedelta(weeks=self.n_weeks)

        self._people()
        self._services()
        self._accounts()
        self._roster()
        self._changes_and_configs()
        self._deploys_and_incidents()
        self._logs()

        files: dict[str, str] = {}
        files["inventory/service-registry.md"] = self._f_registry()
        files["people/directory.md"] = self._f_directory()
        files["oncall/roster.md"] = self._f_roster()
        for svc in self.config_svcs:
            files[self.configs[svc]["path"]] = self._f_config(svc)
        files["changes/change-calendar.md"] = self._f_changes()
        for i, path in enumerate(self.deploy_paths):
            files[path] = self._f_deploys(i)
        for inc in self.incident_ids:
            files[self.incidents[inc]["path"]] = self._f_incident(inc)
        for inc in self.pm_incidents:
            files[self.postmortems[inc]["path"]] = self._f_postmortem(inc)
        for svc in self.log_svcs:
            files[self.logs[svc]["path"]] = self._f_log(svc)
        files["db/accounts_dump.sql"] = self._f_accounts()
        files["finance/vendor-invoices.csv"] = self._f_invoices()
        self._runbooks_facts()
        for key in ("db-failover", "cache-eviction", "cert-rotation", "queue-backlog"):
            files[self.runbooks[key]["path"]] = self._f_runbook(key)
        files["docs/architecture.md"] = self._f_architecture()

        order = list(files)
        rng.shuffle(order)

        facts = {
            "company": self.company,
            "slug": self.slug,
            "window_start": _fmt_ts(self.window_start),
            "people": self.people,
            "services": self.services,
            "configs": self.configs,
            "changes": self.changes,
            "deploys": self.deploys,
            "deploy_by_id": {d["deploy_id"]: d for d in self.deploys},
            "incidents": self.incidents,
            "roster": self.roster,
            "logs": self.logs,
            "accounts": self.accounts,
            "invoices": self.invoices,
            "runbooks": self.runbooks,
            "postmortems": self.postmortems,
            "architecture": self.architecture,
        }
        return World(seed=self.seed, company=self.company, files=files,
                     reading_order=order, facts=facts)

    # -- entities ------------------------------------------------------------
    def _people(self):
        rng = self.rng
        firsts = rng.sample(FIRST, 15)
        firsts.insert(1, firsts[0])  # two people share a first name (distractor)
        lasts = rng.sample(LAST, 16)
        self.people = []
        for f, l in zip(firsts, lasts):
            handle = (f[0] + l).lower()
            self.used.add(handle)
            self.people.append({
                "name": f"{f} {l}",
                "handle": handle,
                "employee_id": self.uniq(lambda: f"E{rng.randint(10000, 99999)}"),
                "team": rng.choice(TEAMS),
                "title": rng.choice(TITLES),
                "location": rng.choice(LOCATIONS),
            })
        # Former staff: reuse first names, new surnames, different ids.
        spare_last = [x for x in LAST if x not in lasts]
        self.former = []
        for f in rng.sample(firsts, 4):
            l = rng.choice(spare_last)
            self.former.append({
                "name": f"{f} {l}",
                "handle": (f[0] + l).lower() + "-x",
                "employee_id": self.uniq(lambda: f"E{rng.randint(10000, 99999)}"),
                "left": (self.window_start - timedelta(days=rng.randint(40, 400))).date().isoformat(),
            })
        self.by_handle = {p["handle"]: p for p in self.people}

    def _services(self):
        rng = self.rng
        pairs = rng.sample(SERVICE_PAIRS, 6)
        self.services = {}
        for a, b in pairs:
            pa = self.uniq(lambda: rng.randint(8010, 9899))
            s = str(pa)
            swapped = int(s[:2] + s[3] + s[2])
            pb = swapped if str(swapped) not in self.used and swapped != pa else None
            if pb is None:
                pb = self.uniq(lambda: rng.randint(8010, 9899))
            else:
                self.used.add(str(pb))
            for name, port in ((a, pa), (b, pb)):
                self.services[name] = {
                    "port": port,
                    "team": rng.choice(TEAMS),
                    "lead": rng.choice(self.people)["handle"],
                    "tier": rng.choice(["tier-1", "tier-1", "tier-2", "tier-3"]),
                    "repo": f"git.{self.slug}.internal/{name}",
                    "prefix": "".join(w[0] for w in name.split("-")).upper() + name[1].upper(),
                }
        names = list(self.services)
        self.svc_names = names
        self.config_svcs = rng.sample(names, 6)
        self.log_svcs = rng.sample(names, 6)
        self.incident_svcs = rng.sample(names, 6)

    def _accounts(self):
        rng = self.rng
        self.accounts = []
        names_used: list[str] = []
        for _ in range(22):
            if names_used and rng.random() < 0.25:
                base = rng.choice(names_used).split(" ")
                mid = rng.choice([m for m in ACC_MID if m != base[1]])
                name = " ".join([base[0], mid] + base[2:])
            else:
                name = f"{rng.choice(ACC_ADJ)} {rng.choice(ACC_MID)} {rng.choice(ACC_SUF)}"
            if name in names_used:
                continue
            names_used.append(name)
            self.accounts.append({
                "account_id": self.uniq(lambda: f"ACC-{rng.randint(10000, 99999)}"),
                "company": name,
                "plan": rng.choice(["starter", "growth", "growth", "enterprise"]),
                "mrr_usd": self.uniq(lambda: f"{rng.randint(90, 18000)}.{rng.randint(0, 99):02d}"),
                "region": rng.choice(["eu-west-1", "us-east-1", "us-west-2", "ap-south-1"]),
                "created_at": (self.window_start - timedelta(days=rng.randint(30, 900))).date().isoformat(),
                "csm": rng.choice(self.people)["handle"],
            })

    def _roster(self):
        rng = self.rng
        pool = [p for p in self.people]
        self.roster = []
        prev = None
        for w in range(self.n_weeks):
            start = self.window_start + timedelta(weeks=w)
            choices = [p for p in pool if p["name"] != prev]
            prim, sec = rng.sample(choices, 2)
            prev = prim["name"]
            self.roster.append({
                "week": f"W{w + 1}",
                "start": _fmt_ts(start),
                "end": _fmt_ts(start + timedelta(weeks=1)),
                "primary": prim["name"],
                "secondary": sec["name"],
            })

    def oncall_at(self, dt: datetime) -> dict:
        for r in self.roster:
            s = datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%SZ")
            e = datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%SZ")
            if s <= dt < e:
                return r
        raise ValueError(dt)

    def _changes_and_configs(self):
        rng = self.rng
        self.changes = {}
        self.configs = {}
        for svc in self.config_svcs:
            port = self.services[svc]["port"]
            old = self.uniq(lambda: rng.randint(8010, 9899))
            chg = self.uniq(lambda: f"CHG-{rng.randint(1000, 9999)}")
            when = self.rand_dt(self.window_start - timedelta(days=60), self.window_start - timedelta(days=5))
            self.changes[chg] = {
                "change_id": chg,
                "date": when.date().isoformat(),
                "service": svc,
                "summary": f"Move production listen port {old} -> {port}",
                "requested_by": rng.choice(self.people)["name"],
                "approved_by": rng.choice(self.people)["name"],
            }
            self.configs[svc] = {
                "path": f"config/{svc}.yaml",
                "port": port,
                "old_port": old,
                "change_id": chg,
                "timeout_ms": self.uniq(lambda: rng.randrange(800, 9000, 50)),
                "pool_size": rng.randint(8, 96),
                "staging_port": self.uniq(lambda: rng.randint(10010, 19899)),
                "staging_timeout_ms": self.uniq(lambda: rng.randrange(800, 9000, 50)),
                "db_host": f"pg-{svc.split('-')[0]}-{rng.randint(1, 4):02d}.prod.internal",
                "max_inflight": rng.randint(100, 4000),
            }
        # Unrelated changes (distractors).
        kinds = ["Rotate TLS certificates", "Increase connection pool", "Enable gzip on responses",
                 "Bump base image", "Lower log verbosity", "Add read replica",
                 "Migrate cron job", "Resize node group"]
        for _ in range(rng.randint(9, 12)):
            chg = self.uniq(lambda: f"CHG-{rng.randint(1000, 9999)}")
            when = self.rand_dt(self.window_start - timedelta(days=60), self.window_end)
            self.changes[chg] = {
                "change_id": chg,
                "date": when.date().isoformat(),
                "service": rng.choice(self.svc_names),
                "summary": rng.choice(kinds),
                "requested_by": rng.choice(self.people)["name"],
                "approved_by": rng.choice(self.people)["name"],
            }
        self.change_order = sorted(self.changes, key=lambda c: (self.changes[c]["date"], c))

    def _deploys_and_incidents(self):
        rng = self.rng
        handles = [p["handle"] for p in self.people]
        raw = []  # (dt, svc, status, tag)
        self.incidents = {}
        self.incident_ids = []
        for svc in self.incident_svcs:
            while True:
                start = self.rand_dt(self.window_start + timedelta(days=2),
                                     self.window_end - timedelta(days=1))
                if start.weekday() == 0 and 6 <= start.hour <= 12:
                    continue  # stay clear of Monday handovers
                break
            inc = self.uniq(lambda: f"INC-{rng.randint(1000, 9999)}")
            dep_dt = start - timedelta(minutes=rng.randint(25, 95), seconds=rng.randint(0, 59))
            detected = start + timedelta(minutes=rng.randint(6, 40), seconds=rng.randint(0, 59))
            resolved = detected + timedelta(minutes=rng.randint(30, 200))
            resolver, reporter = rng.sample(self.people, 2)
            self.incident_ids.append(inc)
            self.incidents[inc] = {
                "incident_id": inc,
                "path": f"tickets/{inc}.md",
                "service": svc,
                "port": self.services[svc]["port"],
                "severity": rng.choice(["SEV1", "SEV2", "SEV2", "SEV3"]),
                "impact_start": _fmt_ts(start),
                "detected": _fmt_ts(detected),
                "resolved": _fmt_ts(resolved),
                "resolved_by": resolver["handle"],
                "reported_by": reporter["handle"],
                "symptom": rng.choice(INCIDENT_SYMPTOMS),
                "failed_requests": rng.randint(1200, 98000),
            }
            raw.append((dep_dt, svc, "ok", inc))
            # rollback after resolution
            raw.append((resolved - timedelta(minutes=rng.randint(5, 25)), svc, "rolled_back", None))
            self.incidents[inc]["oncall"] = self.oncall_at(start)
        self.incident_ids.sort(key=lambda i: self.incidents[i]["impact_start"])
        # Filler deploys
        for _ in range(rng.randint(52, 62)):
            raw.append((self.rand_dt(self.window_start, self.window_end - timedelta(hours=1)),
                        rng.choice(self.svc_names), rng.choice(["ok"] * 8 + ["failed"]), None))
        raw.sort(key=lambda r: r[0])
        version = {s: [rng.randint(1, 4), rng.randint(0, 30), rng.randint(0, 9)] for s in self.svc_names}
        notes_pool = ["canary 10%", "canary 25%", "full rollout", "hotfix", "config only", "",
                      "", "dependency bump", "rebuild", "feature flag default change"]
        self.deploys = []
        mid = self.window_start + timedelta(weeks=self.n_weeks // 2)
        self.deploy_paths = [
            f"deploys/deploy-log-{self.window_start.date().isoformat()}.csv",
            f"deploys/deploy-log-{mid.date().isoformat()}.csv",
        ]
        for dt, svc, status, tag in raw:
            v = version[svc]
            if status == "rolled_back":
                v[2] = max(0, v[2] - 1)
            elif rng.random() < 0.2:
                v[1] += 1
                v[2] = 0
            else:
                v[2] += 1
            d = {
                "deploy_id": self.uniq(lambda: f"D-{rng.randint(10000, 99999)}"),
                "ts": _fmt_ts(dt),
                "service": svc,
                "version": f"{v[0]}.{v[1]}.{v[2]}",
                "commit": self.uniq(lambda: self.hexs(12)),
                "deployer": rng.choice(handles),
                "status": status,
                "notes": "rollback" if status == "rolled_back" else rng.choice(notes_pool),
                "path": self.deploy_paths[0 if dt < mid else 1],
            }
            self.deploys.append(d)
            if tag:
                self.incidents[tag]["trigger_deploy"] = d["deploy_id"]
                self.incidents[tag]["trigger_commit"] = d["commit"]
                self.incidents[tag]["trigger_version"] = d["version"]
                self.incidents[tag]["trigger_path"] = d["path"]
        self.pm_incidents = rng.sample(self.incident_ids, 2)
        self.postmortems = {}
        actions = ["Add alert on connection refusals", "Write regression test for config loader",
                   "Document rollback steps in runbook", "Add canary analysis gate",
                   "Raise default pool limits", "Audit retry settings across clients",
                   "Add dashboard panel for upstream latency", "Review deploy freeze policy"]
        for inc in self.pm_incidents:
            info = self.incidents[inc]
            owners = rng.sample([h for h in handles if h != info["resolved_by"]], 4)
            owners.insert(rng.randint(0, 4), info["resolved_by"])
            start = datetime.strptime(info["resolved"], "%Y-%m-%dT%H:%M:%SZ")
            items = []
            for k, (owner, act) in enumerate(zip(owners, rng.sample(actions, 5))):
                due = self.uniq(lambda start=start: (start + timedelta(days=rng.randint(7, 60))).date().isoformat())
                items.append({"id": f"AI-{k + 1}", "action": act, "owner": owner, "due": due,
                              "status": rng.choice(["open", "open", "done", "in progress"])})
            self.postmortems[inc] = {"path": f"postmortems/PM-{inc}.md", "items": items}

    def _logs(self):
        rng = self.rng
        self.logs = {}
        for svc in self.log_svcs:
            day = self.rand_dt(self.window_start, self.window_end - timedelta(days=1)).date()
            start = datetime(day.year, day.month, day.day, rng.randint(0, 14), 0, 0)
            pre = self.services[svc]["prefix"]
            e1 = self.uniq(lambda pre=pre: f"E-{pre}-{rng.randint(1000, 9999)}")
            e2 = self.uniq(lambda pre=pre: f"E-{pre}-{rng.randint(1000, 9999)}")
            a1, a2 = rng.sample(self.accounts, 2)
            t1 = start + timedelta(seconds=rng.randint(4000, 7000), microseconds=rng.randint(0, 999) * 1000)
            t2 = t1 + timedelta(seconds=rng.randint(1500, 4000), microseconds=rng.randint(0, 999) * 1000)
            self.logs[svc] = {
                "path": f"logs/{svc}-{day.isoformat()}.log",
                "start": start,
                "first_error": {"ts": _fmt_ts(t1, ms=True), "code": e1, "account": a1["account_id"],
                                "req": self.uniq(lambda: self.hexs(8)), "dt": t1},
                "second_error": {"ts": _fmt_ts(t2, ms=True), "code": e2, "account": a2["account_id"],
                                 "req": self.uniq(lambda: self.hexs(8)), "dt": t2},
            }

    def _runbooks_facts(self):
        rng = self.rng
        cluster = f"pg-{self.slug}-core"
        lag_warn, lag, unreach = rng.sample(range(5, 30), 1)[0], rng.randrange(30, 180, 5), rng.randrange(45, 300, 15)
        while unreach == lag:
            unreach += 15
        pct_warn, pct = rng.randint(60, 79), rng.randint(82, 97)
        serial = ":".join(self.hexs(2).upper() for _ in range(8))
        days, days_warn = rng.randint(14, 45), rng.randint(50, 90)
        backlog = rng.randrange(20000, 95000, 500)
        scale = rng.randrange(5000, 19500, 500)
        self.runbooks = {
            "db-failover": {"path": "runbooks/db-failover.md", "cluster": cluster, "warn": lag_warn,
                            "lag": lag, "unreach": unreach, "mins": rng.randint(2, 6),
                            "escalation": rng.choice(self.people)["handle"]},
            "cache-eviction": {"path": "runbooks/cache-eviction.md", "warn": pct_warn, "pct": pct,
                               "cluster": f"redis-{self.slug}-{rng.randint(1, 9)}",
                               "escalation": rng.choice(self.people)["handle"]},
            "cert-rotation": {"path": "runbooks/cert-rotation.md", "serial": serial, "days": days,
                              "days_warn": days_warn, "escalation": rng.choice(self.people)["handle"]},
            "queue-backlog": {"path": "runbooks/queue-backlog.md", "backlog": backlog, "scale": scale,
                              "queue": f"{self.slug}.dispatch.v{rng.randint(2, 5)}",
                              "escalation": rng.choice(self.people)["handle"]},
        }

    # -- file renderers --------------------------------------------------------
    def _f_registry(self) -> str:
        rng = self.rng
        lines = [f"# {self.company} — service registry", "",
                 "Authoritative list of production services. Ports are the production listen ports.", "",
                 "| Service | Port | Team | Tech lead | Tier | Repository |",
                 "|---|---|---|---|---|---|"]
        for name in sorted(self.services):
            s = self.services[name]
            lines.append(f"| {name} | {s['port']} | {s['team']} | @{s['lead']} | {s['tier']} | {s['repo']} |")
        head = "\n".join(lines) + "\n\n## Notes\n\n"

        def filler():
            svc = rng.choice(self.svc_names)
            return f"- {svc}: " + self.prose(1)
        return self.pad(head, filler, self.target(2000, 3400))

    def _f_directory(self) -> str:
        rng = self.rng
        lines = [f"# People directory — {self.company} engineering", "",
                 "| Name | Handle | Employee ID | Team | Title | Location |", "|---|---|---|---|---|---|"]
        for p in self.people:
            lines.append(f"| {p['name']} | @{p['handle']} | {p['employee_id']} | {p['team']} | {p['title']} | {p['location']} |")
        lines += ["", "## Former staff (accounts disabled)", "", "| Name | Handle | Employee ID | Left |", "|---|---|---|---|"]
        for p in self.former:
            lines.append(f"| {p['name']} | @{p['handle']} | {p['employee_id']} | {p['left']} |")
        head = "\n".join(lines) + "\n\n## Team notes\n\n"

        def filler():
            return f"- {rng.choice(TEAMS)}: " + self.prose(1)
        return self.pad(head, filler, self.target(2600, 4000))

    def _f_roster(self) -> str:
        rng = self.rng
        lines = ["# On-call roster (production, primary/secondary)", "",
                 "Shifts hand over on Mondays at 09:00 UTC. A shift covers [From, To).", "",
                 "| Week | From (09:00 UTC) | To (09:00 UTC) | Primary | Secondary |", "|---|---|---|---|---|"]
        for r in self.roster:
            lines.append(f"| {r['week']} | {r['start'][:10]} | {r['end'][:10]} | {r['primary']} | {r['secondary']} |")
        lines += ["", "## Previous rotation (archived, for reference only)", "",
                  "| From | To | Primary | Secondary |", "|---|---|---|---|"]
        for w in range(4, 0, -1):
            s = self.window_start - timedelta(weeks=w)
            a, b = rng.sample(self.people + self.former, 2)
            lines.append(f"| {s.date()} | {(s + timedelta(weeks=1)).date()} | {a['name']} | {b['name']} |")
        head = "\n".join(lines) + "\n\n## Notes\n\n"

        def filler():
            return "- " + self.prose(1)
        return self.pad(head, filler, self.target(2200, 3300))

    def _f_config(self, svc: str) -> str:
        rng = self.rng
        c = self.configs[svc]
        lead = self.services[svc]["lead"]
        flags = ["strict_tls", "async_writes", "shadow_reads", "new_retry_policy", "compress_payloads",
                 "legacy_auth_fallback", "batch_ack", "region_pinning"]
        lines = [
            f"# config/{svc}.yaml — production configuration",
            f"# Owner: @{lead}. Values under `staging:` are NOT used in production.",
            f"service: {svc}",
            "environment: production",
            "listen:",
            "  host: 0.0.0.0",
            f"  port: {c['port']}            # was {c['old_port']} until {c['change_id']}",
            f"  max_inflight_requests: {c['max_inflight']}",
            "database:",
            f"  host: {c['db_host']}",
            f"  pool_size: {c['pool_size']}",
            f"  timeout_ms: {c['timeout_ms']}",
            f"  statement_cache: {rng.choice(['true', 'false'])}",
            "feature_flags:",
        ]
        for f in rng.sample(flags, 5):
            lines.append(f"  {f}: {rng.choice(['true', 'false'])}")
        lines += [
            "staging:",
            f"  port: {c['staging_port']}",
            f"  timeout_ms: {c['staging_timeout_ms']}",
            f"  pool_size: {rng.randint(2, 8)}",
            "",
            "# ---- changelog (most recent last) ----",
        ]
        head = "\n".join(lines) + "\n"
        tail = f"# {c['change_id']}: listen port {c['old_port']} -> {c['port']} (current)\n"

        values = {
            "retry.max_attempts": ["2", "3", "4", "5"],
            "http.idle_timeout_s": ["30", "60", "90", "120"],
            "log.level": ["info", "debug", "warn"],
            "metrics.interval_s": ["10", "15", "30", "60"],
            "cache.ttl_s": ["60", "300", "600", "900"],
            "worker.threads": ["4", "8", "12", "16"],
            "queue.prefetch": ["10", "25", "50", "100"],
        }
        entries = []

        def filler():
            k = rng.choice(sorted(values))
            when = self.rand_dt(self.window_start - timedelta(days=200), self.window_start - timedelta(days=70))
            line = f"# {when.date()}: set {k} = {rng.choice(values[k])} — {self.prose(1)}"
            entries.append(line)
            return line
        padded = self.pad(head, filler, self.target(2200, 3700), tail)
        kept = [e for e in entries if e in padded]
        return head + "".join(x + "\n" for x in sorted(kept)) + tail

    def _f_changes(self) -> str:
        lines = ["# Change calendar", "",
                 "Approved production changes. A change is effective from its date.", "",
                 "| Change | Date | Service | Description | Requested by | Approved by |",
                 "|---|---|---|---|---|---|"]
        for cid in self.change_order:
            c = self.changes[cid]
            lines.append(f"| {cid} | {c['date']} | {c['service']} | {c['summary']} | {c['requested_by']} | {c['approved_by']} |")
        head = "\n".join(lines) + "\n\n## Policy notes\n\n"
        return self.pad(head, lambda: "- " + self.prose(1), max(len(head) + 300, self.target(2600, 4100)))

    def _f_deploys(self, part: int) -> str:
        path = self.deploy_paths[part]
        rows = [d for d in self.deploys if d["path"] == path]
        lines = ["deploy_id,timestamp,service,version,commit,deployer,status,notes"]
        for d in rows:
            lines.append(f"{d['deploy_id']},{d['ts']},{d['service']},{d['version']},{d['commit']},{d['deployer']},{d['status']},{d['notes']}")
        return "\n".join(lines) + "\n"

    def _f_incident(self, inc: str) -> str:
        rng = self.rng
        i = self.incidents[inc]
        svc = i["service"]
        start = datetime.strptime(i["impact_start"], "%Y-%m-%dT%H:%M:%SZ")
        det = datetime.strptime(i["detected"], "%Y-%m-%dT%H:%M:%SZ")
        res = datetime.strptime(i["resolved"], "%Y-%m-%dT%H:%M:%SZ")
        head = (
            f"# {inc}: {i['symptom']} on {svc}\n\n"
            "| Field | Value |\n|---|---|\n"
            f"| Severity | {i['severity']} |\n| Status | Resolved |\n| Service | {svc} |\n"
            f"| Impact start | {i['impact_start']} |\n| Detected (ticket opened) | {i['detected']} |\n"
            f"| Resolved | {i['resolved']} |\n| Reported by | @{i['reported_by']} |\n"
            f"| Resolved by | @{i['resolved_by']} |\n\n"
            "## Summary\n\n"
            f"Beginning at {start:%H:%M:%S}Z, {svc} started refusing connections on port {i['port']}; "
            f"roughly {i['failed_requests']} requests failed before mitigation. "
            f"The timeline correlates with deploy {i['trigger_deploy']} ({svc} {i['trigger_version']}), "
            "which is considered the trigger. Mitigated by rolling back.\n\n"
            "## Timeline\n\n"
            f"- {start:%H:%M}Z impact begins (reconstructed from load balancer logs)\n"
            f"- {det:%H:%M}Z alert fires, ticket opened by @{i['reported_by']}\n"
        )
        tail = f"- {res:%H:%M}Z resolved by @{i['resolved_by']} after rollback; monitoring\n"
        handles = [p["handle"] for p in self.people]
        comments = ["checked dashboards, error budget burn is significant",
                    "customer support has been notified", "no data loss observed so far",
                    "pausing the deploy pipeline for this service", "upstream dependencies look healthy",
                    "database metrics are normal", "will follow up with a postmortem if needed",
                    "status page updated", "paging the service owner"]

        def filler():
            return f"- (comment) @{rng.choice(handles)}: {rng.choice(comments)}. {self.prose(1)}"
        body = self.pad(head, filler, self.target(2300, 3900) - 200, tail)
        return body + "\n## Follow-up\n\n" + self.prose(2) + "\n"

    def _f_postmortem(self, inc: str) -> str:
        pm = self.postmortems[inc]
        i = self.incidents[inc]
        lines = [f"# Postmortem: {inc} ({i['service']})", "",
                 "Blameless review. See the ticket for the detailed timeline.", "",
                 "## What went wrong", "", self.prose(3), "",
                 "## Action items", "", "| ID | Action | Owner | Due | Status |", "|---|---|---|---|---|"]
        for it in pm["items"]:
            lines.append(f"| {it['id']} | {it['action']} | @{it['owner']} | {it['due']} | {it['status']} |")
        head = "\n".join(lines) + "\n\n## Discussion notes\n\n"
        return self.pad(head, lambda: "- " + self.prose(2), self.target(2200, 3600))

    def _f_log(self, svc: str) -> str:
        rng = self.rng
        lg = self.logs[svc]
        start = lg["start"]
        target = self.target(2800, 4100)
        paths = ["/v1/orders", "/v1/routes", "/v2/quotes", "/healthz", "/v1/shipments", "/v1/tokens"]
        entries = []
        e1, e2 = lg["first_error"], lg["second_error"]
        planted = [
            (e1["dt"], f"{e1['ts']} ERROR [{svc}] {e1['code']} upstream write failed account={e1['account']} req={e1['req']}"),
            (e2["dt"], f"{e2['ts']} ERROR [{svc}] {e2['code']} handler panic recovered account={e2['account']} req={e2['req']}"),
        ]
        size = sum(len(x[1]) + 1 for x in planted)
        end = e2["dt"] + timedelta(seconds=1800)
        while True:
            dt = self.rand_dt(start, end) + timedelta(microseconds=rng.randint(0, 999) * 1000)
            ts = _fmt_ts(dt, ms=True)
            kind = rng.random()
            if kind < 0.6:
                line = (f"{ts} INFO  [{svc}] req={self.hexs(8)} {rng.choice(['GET', 'POST', 'PUT'])} "
                        f"{rng.choice(paths)} {rng.choice([200, 200, 200, 201, 204, 404])} {rng.randint(3, 900)}ms")
            elif kind < 0.85:
                line = f"{ts} DEBUG [{svc}] cache hit_ratio={rng.random():.2f} shard={rng.randint(0, 15)}"
            else:
                line = (f"{ts} WARN  [{svc}] W-{rng.randint(100, 999)} slow upstream "
                        f"{rng.choice(self.svc_names)} took {rng.randint(900, 5000)}ms")
            if size + len(line) + 1 > target:
                break
            entries.append((dt, line))
            size += len(line) + 1
        entries += planted
        entries.sort(key=lambda x: x[0])
        return "\n".join(x[1] for x in entries) + "\n"

    def _f_accounts(self) -> str:
        lines = ["-- pg_dump excerpt: table public.accounts (production snapshot)",
                 "-- mrr_usd is monthly recurring revenue in USD",
                 "CREATE TABLE accounts (account_id text PRIMARY KEY, company text, plan text,",
                 "  mrr_usd numeric(10,2), region text, created_at date, csm text);", ""]
        for a in self.accounts:
            lines.append(f"INSERT INTO accounts VALUES ('{a['account_id']}', '{a['company']}', '{a['plan']}', "
                         f"{a['mrr_usd']}, '{a['region']}', '{a['created_at']}', '{a['csm']}');")
        return "\n".join(lines) + "\n"

    def _f_invoices(self) -> str:
        rng = self.rng
        self.invoices = []
        lines = ["invoice_no,vendor,po_number,amount,currency,issued,due,status"]
        year = self.window_start.year
        for _ in range(rng.randint(28, 34)):
            issued = self.rand_dt(self.window_start - timedelta(days=60), self.window_end).date()
            inv = {
                "invoice_no": self.uniq(lambda: f"INV-{year}-{rng.randint(10000, 99999)}"),
                "vendor": rng.choice(VENDORS),
                "po_number": self.uniq(lambda: f"PO-{rng.randint(10000, 99999)}"),
                "amount": self.uniq(lambda: f"{rng.randint(120, 48000)}.{rng.randint(0, 99):02d}"),
                "currency": rng.choice(["USD", "USD", "EUR"]),
                "issued": issued.isoformat(),
                "due": (issued + timedelta(days=rng.choice([15, 30, 45]))).isoformat(),
                "status": rng.choice(["paid", "paid", "unpaid", "disputed"]),
            }
            self.invoices.append(inv)
            lines.append(",".join(str(inv[k]) for k in ("invoice_no", "vendor", "po_number", "amount",
                                                         "currency", "issued", "due", "status")))
        return "\n".join(lines) + "\n"

    def _f_runbook(self, key: str) -> str:
        r = self.runbooks[key]
        if key == "db-failover":
            head = (f"# Runbook: Postgres primary failover ({r['cluster']})\n\n"
                    f"Escalation: @{r['escalation']}\n\n## When to use\n\n"
                    f"A warning alert fires at {r['warn']} seconds of replication lag; it is informational only.\n\n"
                    f"**Failover threshold:** initiate failover when replication lag exceeds {r['lag']} seconds "
                    f"for {r['mins']} consecutive minutes, or when the primary has been unreachable for more "
                    f"than {r['unreach']} seconds.\n\n## Procedure\n\n"
                    f"1. Confirm lag on the replication dashboard for {r['cluster']}.\n"
                    f"2. Freeze writes: `pgctl freeze --cluster {r['cluster']}`.\n"
                    f"3. Promote the healthiest replica: `pgctl promote --cluster {r['cluster']} --replica auto`.\n"
                    "4. Repoint the write DNS record and verify connections drain.\n\n## Notes\n\n")
        elif key == "cache-eviction":
            head = (f"# Runbook: cache memory pressure ({r['cluster']})\n\n"
                    f"Escalation: @{r['escalation']}\n\n## Thresholds\n\n"
                    f"- Warning at {r['warn']}% memory usage: watch only, no action.\n"
                    f"- **Flush threshold:** run a controlled eviction when memory usage exceeds {r['pct']}%.\n\n"
                    "## Procedure\n\n"
                    f"1. Check `redis-cli -h {r['cluster']} info memory`.\n"
                    "2. Evict cold keyspaces first (`sessions:*` before `routes:*`).\n"
                    "3. Never run FLUSHALL in production.\n\n## Notes\n\n")
        elif key == "cert-rotation":
            head = (f"# Runbook: TLS certificate rotation (edge wildcard)\n\n"
                    f"Escalation: @{r['escalation']}\n\n## Current certificate\n\n"
                    f"- Serial: {r['serial']}\n- Issuer: internal ACME (prod)\n\n## Thresholds\n\n"
                    f"- Informational reminder at {r['days_warn']} days before expiry.\n"
                    f"- **Rotation threshold:** rotate when the certificate expires within {r['days']} days.\n\n"
                    "## Procedure\n\n1. Request a new certificate via the ACME job.\n"
                    "2. Stage it on one edge node and verify the chain.\n3. Roll to all edge nodes.\n\n## Notes\n\n")
        else:
            head = (f"# Runbook: dispatch queue backlog ({r['queue']})\n\n"
                    f"Escalation: @{r['escalation']}\n\n## Thresholds\n\n"
                    f"- Autoscaling adds consumers when backlog exceeds {r['scale']} messages (automatic).\n"
                    f"- **Paging threshold:** page the on-call primary if backlog exceeds {r['backlog']} messages "
                    "for 10 minutes.\n\n## Procedure\n\n"
                    "1. Check consumer lag per partition.\n2. Look for a poison message in the dead-letter topic.\n"
                    "3. Scale consumers manually if autoscaling is saturated.\n\n## Notes\n\n")
        return self.pad(head, lambda: "- " + self.prose(2), self.target(2400, 3800))

    def _f_architecture(self) -> str:
        rng = self.rng
        topics = ["shipments.events", "routes.updates", "billing.ledger", "notifications.out", "audit.trail"]
        parts = rng.sample([12, 16, 24, 32, 48, 64, 96, 128], len(topics))
        bus = f"kafka-{self.slug}-0{rng.randint(1, 9)}"
        self.architecture = {"path": "docs/architecture.md", "bus": bus,
                             "partitions": dict(zip(topics, parts))}
        lines = [f"# {self.company} platform architecture (overview)", "", self.prose(3), "",
                 "## Message bus", "", f"All asynchronous traffic goes through the {bus} cluster.", "",
                 "| Topic | Partitions | Retention |", "|---|---|---|"]
        for t, p in zip(topics, parts):
            lines.append(f"| {t} | {p} | {rng.choice([24, 72, 168])}h |")
        head = "\n".join(lines) + "\n\n## Services\n\n"

        def filler():
            return f"- {rng.choice(self.svc_names)}: {self.prose(1)}"
        return self.pad(head, filler, self.target(2600, 4000))


def generate_world(seed: int) -> World:
    """Build the world for ``seed``. Pure function of the seed."""
    return _Builder(seed).build()
