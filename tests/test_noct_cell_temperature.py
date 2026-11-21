"""Tests for the NOCT cell temperature model and the derate that uses it.

The helpers (NOCT_C, WIND_HEIGHT_FACTOR, _wind_factor, _cell_temp_c,
_temp_derate) live as copies in each script, because each script also runs
as a standalone notebook. These tests pull the helper source out of each .py
file with ast, without importing the scripts (they make network calls at
import), and check:

  - every copy computes the NOCT cell temperature with the SAM wind term,
    T_cell = T_air + G / 800 * (NOCT - 20) * 9.5 / (5.7 + 3.8 * 0.51 * v10),
    with a wind factor of 1.0 when wind is unknown, and
    derate = max(0.75, 1.0 - 0.004 * max(0, T_cell - 25)),
  - every copy uses NOCT_C = 49.0 and the copies are identical,
  - each matching notebook holds the same helper source,
  - get_solar_production keeps its signature and return keys, and uses the
    NOCT derate on both the GHI path and the cloud cover fallback path,
  - web-litert/app.js gives the same values (when node is installed),
  - no live file still applies the old ambient-only derate.

Expected values are computed from the formula inside the tests.

Run from repo root:
    python -m pytest tests/test_noct_cell_temperature.py -v
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "solarhive_datagen",
    "solarhive_finetune",
    "solarhive_inference",
    "solarhive_e4b_litert_v3.1",
]
TOOL_SCRIPTS = ["solarhive_inference", "solarhive_e4b_litert_v3.1"]
HELPER_NAMES = ("NOCT_C", "WIND_HEIGHT_FACTOR", "_wind_factor", "_cell_temp_c",
                "_temp_derate")

NOCT_EXPECTED = 49.0
AIR_TEMPS_C = [-10.0, 0.0, 20.0, 25.0, 30.0, 38.0, 45.0, 54.4]
IRRADIANCES = [0.0, 100.0, 250.0, 500.0, 800.0, 1000.0, 1200.0, 1400.0]
WINDS_MS = [None, 0.0, 1.0, 1.0 / 0.51, 3.6, 8.0, 15.0]


def expected_wind_factor(v10):
    # SAM Technical Reference (NREL/TP-6A20-67399) Eqs. 10.35 and 10.37
    if v10 is None:
        return 1.0
    return 9.5 / (5.7 + 3.8 * 0.51 * max(0.0, v10))


def expected_cell_temp(temp_air_c, g, v10=None, noct=NOCT_EXPECTED):
    return temp_air_c + max(0.0, g) / 800.0 * (noct - 20.0) * expected_wind_factor(v10)


def expected_derate(temp_air_c, g, v10=None, noct=NOCT_EXPECTED):
    t_cell = expected_cell_temp(temp_air_c, g, v10, noct)
    return max(0.75, 1.0 - 0.004 * max(0.0, t_cell - 25.0))


def _source(stem: str, suffix: str = ".py") -> str:
    return (REPO_ROOT / f"{stem}{suffix}").read_text(encoding="utf-8")


def _helper_nodes(stem: str):
    """Return {name: ast node} for the helper definitions in a script."""
    tree = ast.parse(_source(stem))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES:
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in HELPER_NAMES:
                    found[target.id] = node
    return found


def load_helpers(stem: str) -> dict:
    nodes = _helper_nodes(stem)
    missing = set(HELPER_NAMES) - set(nodes)
    if missing:
        raise AssertionError(f"{stem}.py is missing {sorted(missing)}")
    module = ast.Module(body=[nodes[n] for n in HELPER_NAMES], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, f"{stem}.py", "exec"), namespace)
    return namespace


def load_function(stem: str, name: str, namespace: dict):
    tree = ast.parse(_source(stem))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), f"{stem}.py", "exec"),
                 namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in {stem}.py")


class TestHelperFormula(unittest.TestCase):
    def test_constants(self):
        for stem in SCRIPTS:
            h = load_helpers(stem)
            with self.subTest(stem=stem):
                self.assertEqual(h["NOCT_C"], NOCT_EXPECTED)
                self.assertEqual(h["WIND_HEIGHT_FACTOR"], 0.51)

    def test_cell_temp_matches_formula(self):
        for stem in SCRIPTS:
            h = load_helpers(stem)
            for t in AIR_TEMPS_C:
                for g in IRRADIANCES:
                    for v in WINDS_MS:
                        with self.subTest(stem=stem, t=t, g=g, v=v):
                            self.assertAlmostEqual(h["_cell_temp_c"](t, g, v),
                                                   expected_cell_temp(t, g, v), places=12)

    def test_derate_matches_formula(self):
        for stem in SCRIPTS:
            h = load_helpers(stem)
            for t in AIR_TEMPS_C:
                for g in IRRADIANCES:
                    for v in WINDS_MS:
                        with self.subTest(stem=stem, t=t, g=g, v=v):
                            self.assertAlmostEqual(h["_temp_derate"](t, g, v),
                                                   expected_derate(t, g, v), places=12)

    def test_unknown_wind_equals_noct_rating_wind(self):
        # Wind factor 1.0 means v_adj = 1 m/s, the NOCT rating wind at module height.
        h = load_helpers("solarhive_inference")
        self.assertEqual(h["_wind_factor"](None), 1.0)
        self.assertEqual(h["_wind_factor"](float("nan")), 1.0)
        self.assertAlmostEqual(h["_wind_factor"](1.0 / 0.51), 1.0, places=12)
        self.assertEqual(h["_wind_factor"](-3.0), h["_wind_factor"](0.0))

    def test_wind_cools_cells(self):
        h = load_helpers("solarhive_inference")
        temps = [h["_cell_temp_c"](30.0, 1000.0, v) for v in (0.0, 2.0, 5.0, 10.0)]
        self.assertEqual(temps, sorted(temps, reverse=True))

    def test_zero_irradiance_gives_air_temperature(self):
        h = load_helpers("solarhive_inference")
        for t in AIR_TEMPS_C:
            for v in WINDS_MS:
                self.assertEqual(h["_cell_temp_c"](t, 0.0, v), t)

    def test_negative_irradiance_treated_as_zero(self):
        h = load_helpers("solarhive_inference")
        self.assertEqual(h["_cell_temp_c"](30.0, -50.0), 30.0)

    def test_noct_conditions_give_noct(self):
        # By definition, 20°C air, 800 W/m² and the rating wind give NOCT.
        h = load_helpers("solarhive_inference")
        self.assertAlmostEqual(h["_cell_temp_c"](20.0, 800.0), NOCT_EXPECTED, places=12)
        self.assertAlmostEqual(h["_cell_temp_c"](20.0, 800.0, 1.0 / 0.51),
                               NOCT_EXPECTED, places=12)

    def test_floor_and_ceiling(self):
        h = load_helpers("solarhive_inference")
        self.assertEqual(h["_temp_derate"](54.4, 1400.0), 0.75)
        self.assertEqual(h["_temp_derate"](0.0, 0.0), 1.0)
        self.assertEqual(h["_temp_derate"](25.0, 0.0), 1.0)

    def test_derate_never_increases_with_irradiance(self):
        h = load_helpers("solarhive_inference")
        for t in AIR_TEMPS_C:
            for v in WINDS_MS:
                values = [h["_temp_derate"](t, g, v) for g in IRRADIANCES]
                self.assertEqual(values, sorted(values, reverse=True))

    def test_all_copies_identical(self):
        def dump(stem):
            nodes = _helper_nodes(stem)
            return ast.dump(ast.Module(body=[nodes[n] for n in HELPER_NAMES],
                                       type_ignores=[]))
        reference = dump(SCRIPTS[0])
        for stem in SCRIPTS[1:]:
            with self.subTest(stem=stem):
                self.assertEqual(dump(stem), reference)


class TestNotebooksMatchScripts(unittest.TestCase):
    def test_notebook_holds_same_helper_source(self):
        for stem in SCRIPTS:
            py_src = _source(stem).replace("\r\n", "\n")
            nb = json.loads(_source(stem, ".ipynb"))
            code = "\n".join("".join(c["source"]) for c in nb["cells"]
                             if c["cell_type"] == "code")
            for name, node in _helper_nodes(stem).items():
                segment = ast.get_source_segment(py_src, node)
                with self.subTest(stem=stem, name=name):
                    self.assertIn(segment, code)

    def test_notebook_tool_matches_script(self):
        for stem in TOOL_SCRIPTS:
            py_src = _source(stem).replace("\r\n", "\n")
            nb = json.loads(_source(stem, ".ipynb"))
            code = "\n".join("".join(c["source"]) for c in nb["cells"]
                             if c["cell_type"] == "code")
            tree = ast.parse(py_src)
            for node in tree.body:
                if (isinstance(node, ast.FunctionDef)
                        and node.name in ("get_solar_production", "_get_current_ghi_and_wind")):
                    with self.subTest(stem=stem, name=node.name):
                        self.assertIn(ast.get_source_segment(py_src, node), code)


class _FixedHour:
    def __init__(self, hour):
        self.hour = hour


def _fake_datetime(hour):
    class _DT:
        @staticmethod
        def now(tz=None):
            return _FixedHour(hour)
    return _DT


class TestGetSolarProduction(unittest.TestCase):
    CAPACITY = 72

    def _tool(self, stem, ghi, hour=12, wind=None):
        ns = load_helpers(stem)
        ns.update({
            "COMMUNITY_CAPACITY_KW": self.CAPACITY,
            "SYSTEM_EFF": 0.85,
            "_get_current_ghi_and_wind": lambda: (ghi, wind),
            "datetime": _fake_datetime(hour),
        })
        return load_function(stem, "get_solar_production", ns)

    def test_signature_unchanged(self):
        for stem in TOOL_SCRIPTS:
            params = inspect.signature(self._tool(stem, ghi=500.0)).parameters
            with self.subTest(stem=stem):
                self.assertEqual(list(params), ["clouds_pct", "temp_f"])
                self.assertEqual(params["clouds_pct"].default, 30)
                self.assertEqual(params["temp_f"].default, 77.0)

    def test_ghi_path_uses_noct_derate(self):
        for stem in TOOL_SCRIPTS:
            for temp_f in (77.0, 100.4):
                for ghi in (0.0, 500.0, 1000.0):
                    for wind in (None, 0.0, 3.6):
                        out = self._tool(stem, ghi=ghi, wind=wind)(clouds_pct=10,
                                                                   temp_f=temp_f)
                        derate = expected_derate((temp_f - 32) * 5 / 9, ghi, wind)
                        with self.subTest(stem=stem, temp_f=temp_f, ghi=ghi, wind=wind):
                            self.assertEqual(out["source"], "open-meteo")
                            self.assertEqual(out["temp_derate_pct"],
                                             round(derate * 100, 1))
                            self.assertEqual(out["production_kw"], round(
                                max(0, self.CAPACITY * (ghi / 1000) * 0.85 * derate), 1))

    def test_fallback_uses_implied_irradiance(self):
        for stem in TOOL_SCRIPTS:
            for clouds, hour in ((0, 12), (50, 10), (100, 15), (20, 22)):
                out = self._tool(stem, ghi=None, hour=hour)(clouds_pct=clouds,
                                                            temp_f=100.4)
                efficiency = max(0.15, 0.85 - (clouds / 100) * 0.70)
                tf = max(0, 1 - ((hour - 12) / 6) ** 2) if 6 <= hour <= 18 else 0
                derate = expected_derate(38.0, 1000 * efficiency * tf / 0.85)
                with self.subTest(stem=stem, clouds=clouds, hour=hour):
                    self.assertEqual(out["source"], "fallback")
                    self.assertAlmostEqual(out["temp_derate_pct"], round(derate * 100, 1))
                    self.assertEqual(out["production_kw"], round(
                        self.CAPACITY * efficiency * tf * derate, 1))

    def test_return_keys_unchanged(self):
        for stem in TOOL_SCRIPTS:
            with self.subTest(stem=stem):
                self.assertEqual(
                    set(self._tool(stem, ghi=400.0, wind=2.0)()),
                    {"production_kw", "capacity_kw", "efficiency_pct", "ghi_wm2",
                     "temp_derate_pct", "source"})
                self.assertEqual(
                    set(self._tool(stem, ghi=None)()),
                    {"production_kw", "capacity_kw", "efficiency_pct",
                     "temp_derate_pct", "source"})


class TestNoAmbientOnlyDerateLeft(unittest.TestCase):
    OLD_PATTERNS = [
        re.compile(r"max\(0,\s*temp_c\s*-\s*25\)"),
        re.compile(r"\(t\s*-\s*32\)\s*\*\s*5\s*/\s*9\s*-\s*25"),
        re.compile(r"\(row\[\\?\"temp_f\\?\"\]\s*-\s*32\)\s*\*\s*5\s*/\s*9\s*-\s*25"),
    ]
    FILES = [f"{s}{ext}" for s in SCRIPTS for ext in (".py", ".ipynb")] + ["web-litert/app.js"]

    def test_old_formula_gone(self):
        for rel in self.FILES:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            for pattern in self.OLD_PATTERNS:
                with self.subTest(file=rel, pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(text))


@unittest.skipUnless(shutil.which("node"), "node not installed")
class TestBrowserHelperMatches(unittest.TestCase):
    def test_js_values_match_formula(self):
        js = (REPO_ROOT / "web-litert" / "app.js").read_text(encoding="utf-8")
        start = js.index("const NOCT_C")
        end = js.index("async function tool_get_solar_production")
        cases = [[t, g, v] for t in AIR_TEMPS_C for g in IRRADIANCES for v in WINDS_MS]
        script = (js[start:end]
                  + f"\nconst cases = {json.dumps(cases)};\n"
                  + "console.log(JSON.stringify(cases.map(([t, g, v]) => "
                    "[cellTempC(t, g, v), tempDerate(t, g, v)])));\n")
        result = subprocess.run(["node", "-e", script], capture_output=True,
                                text=True, check=True, timeout=60)
        values = json.loads(result.stdout)
        self.assertEqual(len(values), len(cases))
        for (t, g, v), (cell, derate) in zip(cases, values):
            with self.subTest(t=t, g=g, v=v):
                self.assertAlmostEqual(cell, expected_cell_temp(t, g, v), places=12)
                self.assertAlmostEqual(derate, expected_derate(t, g, v), places=12)


if __name__ == "__main__":
    unittest.main()
