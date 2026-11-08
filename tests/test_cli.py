from __future__ import annotations

import json

import pytest

from dqguard.cli import _coerce_numeric, build_parser, main
from dqguard.table import Table


def test_bench_no_write(capsys):
    rc = main(["bench", "--no-write", "--datasets", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "full" in out and "p=" in out


def test_bench_writes(tmp_path, capsys):
    rc = main(["bench", "--datasets", "3", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_validate_profile(tmp_path, capsys):
    p = tmp_path / "d.csv"
    p.write_text("id,val\n1,x\n2,y\n3,x\n", encoding="utf-8")
    rc = main(["validate", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "profile:" in out and "rows: 3" in out


def test_validate_with_schema_finds_issue(tmp_path, capsys):
    p = tmp_path / "d.csv"
    p.write_text("id,val\n1,x\n,y\n3,x\n", encoding="utf-8")
    schema = tmp_path / "s.json"
    schema.write_text(json.dumps({"not_null": ["id"], "unique": ["val"]}), encoding="utf-8")
    rc = main(["validate", str(p), "--schema", str(schema)])
    assert rc == 1  # found at least one issue
    out = capsys.readouterr().out
    assert "not_null" in out


def test_validate_save_and_reference(tmp_path, capsys):
    p = tmp_path / "d.csv"
    p.write_text("id,val\n1,10\n2,20\n", encoding="utf-8")
    fp = tmp_path / "fp.json"
    main(["validate", str(p), "--coerce", "--save-fingerprint", str(fp)])
    assert fp.exists()
    data = json.loads(fp.read_text())
    assert "types" in data and "order" in data

    # now a drifted file (val becomes strings) vs saved fingerprint
    p2 = tmp_path / "d2.csv"
    p2.write_text("id,val\n1,a\n2,b\n", encoding="utf-8")
    capsys.readouterr()
    main(["validate", str(p2), "--coerce", "--reference", str(fp)])
    out = capsys.readouterr().out
    assert "schema drift" in out


def test_coerce_numeric():
    t = Table({"a": ["1", "2.5", "x", None]})
    c = _coerce_numeric(t)
    assert c.column("a") == [1, 2.5, "x", None]


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
