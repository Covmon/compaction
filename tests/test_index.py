"""Index data model: ops, validation, render, resolve, coverage, JSON parsing."""

import pytest

from compaction.strategies.index import (
    AUTO_SECTION_TITLE,
    OPS_SCHEMA,
    Index,
    IndexLookupError,
    merge_ranges,
    parse_ops,
    parse_range,
)


def build() -> Index:
    idx = Index()
    rep = idx.apply(
        [
            {"op": "add_section", "title": "Task brief"},
            {"op": "add_subsection", "section": "1", "summary": "Task: audit payments", "ranges": ["m0001"]},
            {"op": "add_section", "title": "Config files"},
            # reference a section added in this batch by its exact title
            {"op": "add_subsection", "section": "Config files", "summary": "Read config/payments.yaml: port, timeout", "ranges": ["m0002-m0003"]},
            # null section -> last section added in this batch
            {"op": "add_subsection", "section": None, "summary": "Read config/auth.yaml: token ttl", "ranges": [["m0004", "m0005"], "m0008"]},
        ]
    )
    assert rep.errors == []
    assert (rep.sections_added, rep.subsections_added) == (2, 3)
    return idx


def test_parse_and_merge_ranges():
    assert parse_range("m0005-m0011") == (5, 11)
    assert parse_range("m0005–m0011") == (5, 11)
    assert parse_range("m0005..m0011") == (5, 11)
    assert parse_range("m0007") == (7, 7)
    assert parse_range(["m0002", "m0004"]) == (2, 4)
    assert parse_range({"start": "m0002", "end": "m0003"}) == (2, 3)
    with pytest.raises(ValueError):
        parse_range("hello")
    assert merge_ranges([(5, 7), (1, 2), (3, 3), (10, 12), (11, 15)]) == [(1, 3), (5, 7), (10, 15)]


def test_ops_render_and_ids():
    idx = build()
    out = idx.render()
    assert out.splitlines() == [
        "# Index",
        "## 1. Task brief  [m0001]",
        "### 1.1 Task: audit payments  [m0001]",
        "## 2. Config files  [m0002–m0008]",
        "### 2.1 Read config/payments.yaml: port, timeout  [m0002–m0003]",
        "### 2.2 Read config/auth.yaml: token ttl  [m0004–m0005, m0008]",
    ]
    assert idx.all_ids() == ["1", "1.1", "2", "2.1", "2.2"]
    # continuity edits + extend
    rep = idx.apply(
        [
            {"op": "edit", "id": "2", "title": "Config + secrets files"},
            {"op": "edit", "id": "2.1", "summary": "Read config/payments.yaml: port 8443, timeout"},
            {"op": "extend", "id": "2.2", "ranges": ["m0009-m0010"]},
        ]
    )
    assert rep.errors == [] and (rep.edits, rep.extends) == (2, 1)
    assert idx.section("2").title == "Config + secrets files"
    assert idx.subsection("2.2").ranges == [(4, 5), (8, 10)]
    assert "port 8443" in idx.render()


def test_validation_skips_bad_ops_and_ranges():
    idx = build()
    rep = idx.apply(
        [
            {"op": "add_subsection", "section": "9", "summary": "x", "ranges": ["m0011"]},  # no section 9
            {"op": "add_subsection", "section": "1", "summary": "x", "ranges": ["m0013-m0012"]},  # reversed
            {"op": "add_subsection", "section": "1", "summary": "", "ranges": ["m0011"]},  # empty summary
            {"op": "add_subsection", "section": "1", "summary": "late", "ranges": ["m0011", "m0099", "zzz"]},
            {"op": "edit", "id": "7.1", "summary": "nope"},
            {"op": "extend", "id": "1.1", "ranges": None},
            {"op": "frobnicate"},
            "not a dict",
            {"op": "add_section", "title": "Empty section"},  # never gets a subsection -> dropped
        ],
        valid=lambda n: 1 <= n <= 20,
    )
    # only "late" survived, with the single valid range m0011
    assert rep.subsections_added == 1 and rep.sections_added == 0
    assert idx.subsection("1.2").ranges == [(11, 11)]
    assert len(rep.errors) >= 9
    assert [s.title for s in idx.sections] == ["Task brief", "Config files"]


def test_resolve_expand_and_unknown_ids():
    idx = build()
    assert idx.resolve(["2.2"]) == ["m0004", "m0005", "m0008"]
    assert idx.resolve(["2", "1.1", "2.1"]) == ["m0001", "m0002", "m0003", "m0004", "m0005", "m0008"]
    assert idx.resolve(["m0010-m0012", "m0003"]) == ["m0003", "m0010", "m0011", "m0012"]
    groups = idx.expand(["2"])
    assert [g[0].split()[0] for g in groups] == ["2.1", "2.2"]
    with pytest.raises(IndexLookupError) as e:
        idx.resolve(["3.1", "2.1"])
    assert "3.1" in str(e.value) and "Valid ids: 1, 1.1, 2, 2.1, 2.2" in str(e.value)


def test_coverage_and_auto_cover():
    idx = build()
    cov, unc = idx.coverage(["m0001", "m0006", "m0007", "m0008", "m0020"])
    assert cov == ["m0001", "m0008"] and unc == ["m0006", "m0007", "m0020"]
    n = idx.auto_cover([6, 7, 8, 20, 21, 22, 23, 24, 25, 26, 27], lambda g: f"msgs {g[0]}-{g[-1]}")
    assert n == 10  # 8 was already covered
    auto = idx.sections[-1]
    assert auto.title == AUTO_SECTION_TITLE
    assert [x.ranges for x in auto.subsections] == [[(6, 7)], [(20, 25)], [(26, 27)]]  # max 6 per run
    assert all(x.summary.startswith("(auto)") for x in auto.subsections)
    assert idx.coverage([f"m{i:04d}" for i in range(1, 9)])[1] == []
    # a second auto_cover reuses the auto section
    idx.auto_cover([30], lambda g: "x")
    assert sum(s.title == AUTO_SECTION_TITLE for s in idx.sections) == 1


def test_json_roundtrip_and_empty_render():
    idx = build()
    again = Index.from_json(idx.to_json())
    assert again.render() == idx.render()
    assert Index().render().startswith("# Index\n(empty")
    assert not Index() and idx


def test_parse_ops_fallbacks():
    ops = [{"op": "add_section", "title": "A"}]
    assert parse_ops('{"ops": [{"op": "add_section", "title": "A"}]}') == ops
    assert parse_ops('Sure!\n```json\n{"ops": [{"op": "add_section", "title": "A"}]}\n```\nDone.') == ops
    assert parse_ops('Here you go: {"ops": [{"op": "add_section", "title": "A"}]} hope it helps') == ops
    assert parse_ops('[{"op": "add_section", "title": "A"}]') == ops
    assert parse_ops("I could not index this.") is None
    assert parse_ops('{"ops": [') is None
    assert parse_ops("") is None


def test_ops_schema_is_structured_output_compatible():
    """Every object: additionalProperties=false and all properties required; no
    unsupported keywords."""
    banned = {"minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "prefixItems"}

    def walk(node):
        if isinstance(node, dict):
            assert not banned & node.keys()
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(OPS_SCHEMA)
