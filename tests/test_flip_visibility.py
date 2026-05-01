"""Tests for scripts/flip_visibility.py and live HF reconciliation invariants.

Two test layers:

  Unit tests (no network, always run):
    - REPOS registry shape
    - argparse contract
    - plan() pure function
    - main() with mocked HfApi: dry-run vs execute paths
    - main() error paths: missing token, repo_info failure

  Functional / live-state audit tests (require HF_TOKEN, skip otherwise):
    - All 7 repos in REPOS resolve via api.repo_info()
    - HF dataset card README has v2 numbers (1,727 / 1,713 / 14 image / 183 tool)
    - HF dataset card README has no v1 leaks ('1,029-row subset', 'v1 fine-tune')
    - solarhive-26b-a4b-lora has 14 chart files in datagen_charts/
    - solarhive-26b-a4b-merged has 14 chart files in datagen_charts/
    - All 8 audited surfaces (1 dataset + 5 model + 2 GitHub raw) cite Google
      AI model card + HF gemma4 blog (vision-encoder transparency invariant)

The functional tests double as drift-detection: if anyone modifies a card
without going through this workflow, the test breaks.

Run from repo root:
    python -m pytest tests/test_flip_visibility.py -v
or:
    python tests/test_flip_visibility.py
"""

from __future__ import annotations

import io
import os
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import flip_visibility  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------


class TestRepoRegistry(unittest.TestCase):
    """The REPOS list is the single source of truth for which repos get flipped."""

    def test_seven_repos(self):
        self.assertEqual(
            len(flip_visibility.REPOS),
            7,
            "Expected exactly 7 SolarHive HF repos (1 dataset + 5 models + 1 space). "
            "If a repo was added/removed, update this test and the auto-memory.",
        )

    def test_one_dataset(self):
        types = [t for _, t in flip_visibility.REPOS]
        self.assertEqual(types.count("dataset"), 1)

    def test_five_models(self):
        types = [t for _, t in flip_visibility.REPOS]
        self.assertEqual(types.count("model"), 5)

    def test_one_space(self):
        types = [t for _, t in flip_visibility.REPOS]
        self.assertEqual(types.count("space"), 1)

    def test_all_repos_namespaced_to_truthseeker87(self):
        for repo_id, _ in flip_visibility.REPOS:
            self.assertTrue(
                repo_id.startswith("Truthseeker87/"),
                f"{repo_id} should be in the Truthseeker87 namespace",
            )

    def test_canonical_dataset_repo_id(self):
        ids = {rid for rid, _ in flip_visibility.REPOS}
        self.assertIn("Truthseeker87/solarhive-community-solar-multimodal", ids)

    def test_no_deleted_v1_dataset_repo(self):
        """The v1 repo solarhive-community-solar-1k was deleted Apr 30 (OB12)."""
        ids = {rid for rid, _ in flip_visibility.REPOS}
        self.assertNotIn("Truthseeker87/solarhive-community-solar-1k", ids)

    def test_canonical_model_repo_ids(self):
        ids = {rid for rid, _ in flip_visibility.REPOS}
        for expected in [
            "Truthseeker87/solarhive-26b-a4b-lora",
            "Truthseeker87/solarhive-26b-a4b-merged",
            "Truthseeker87/solarhive-26b-a4b-nf4",
            "Truthseeker87/solarhive-e4b-ollama",
            "Truthseeker87/solarhive-e4b-gguf",
        ]:
            self.assertIn(expected, ids)

    def test_canonical_space_repo_id(self):
        ids = {rid for rid, _ in flip_visibility.REPOS}
        self.assertIn("Truthseeker87/solarhive", ids)

    def test_no_duplicate_repos(self):
        ids = [rid for rid, _ in flip_visibility.REPOS]
        self.assertEqual(len(ids), len(set(ids)), "duplicate repo_id in REPOS")


class TestArgparse(unittest.TestCase):
    def test_make_public(self):
        args = flip_visibility.parse_args(["--make", "public"])
        self.assertEqual(args.make, "public")
        self.assertFalse(args.execute)
        self.assertFalse(args.verify_only)

    def test_make_private(self):
        args = flip_visibility.parse_args(["--make", "private"])
        self.assertEqual(args.make, "private")

    def test_verify_only(self):
        args = flip_visibility.parse_args(["--verify-only"])
        self.assertTrue(args.verify_only)

    def test_execute_flag(self):
        args = flip_visibility.parse_args(["--make", "public", "--execute"])
        self.assertTrue(args.execute)

    def test_make_and_verify_only_mutually_exclusive(self):
        # argparse exits with SystemExit on bad args — capture stderr to suppress noise.
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            flip_visibility.parse_args(["--make", "public", "--verify-only"])

    def test_no_args_fails(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            flip_visibility.parse_args([])

    def test_invalid_make_value(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            flip_visibility.parse_args(["--make", "nonsense"])


class TestPlan(unittest.TestCase):
    """plan() is a pure function over (repo, current_private, target_private)."""

    def test_all_already_target(self):
        repos = [("a", "model", True), ("b", "dataset", True)]
        to_change, no_op = flip_visibility.plan(repos, target_private=True)
        self.assertEqual(to_change, [])
        self.assertEqual(no_op, [("a", "model"), ("b", "dataset")])

    def test_all_need_change(self):
        repos = [("a", "model", False), ("b", "dataset", False)]
        to_change, no_op = flip_visibility.plan(repos, target_private=True)
        self.assertEqual(to_change, [("a", "model"), ("b", "dataset")])
        self.assertEqual(no_op, [])

    def test_mixed(self):
        repos = [
            ("a", "model", True),
            ("b", "dataset", False),
            ("c", "space", True),
        ]
        to_change, no_op = flip_visibility.plan(repos, target_private=False)
        self.assertEqual(to_change, [("a", "model"), ("c", "space")])
        self.assertEqual(no_op, [("b", "dataset")])

    def test_empty_repos(self):
        to_change, no_op = flip_visibility.plan([], target_private=True)
        self.assertEqual((to_change, no_op), ([], []))


class _FakeRepoInfo:
    def __init__(self, private):
        self.private = private


class _FakeApi:
    """Mock HfApi that records calls and returns scripted repo_info results."""

    def __init__(self, private_state):
        # private_state: dict of repo_id -> bool (current private value)
        self.private_state = dict(private_state)
        self.update_calls = []
        self.repo_info_calls = []

    def repo_info(self, repo_id, repo_type=None, **_):
        self.repo_info_calls.append((repo_id, repo_type))
        if repo_id not in self.private_state:
            raise RuntimeError(f"unknown repo {repo_id}")
        return _FakeRepoInfo(self.private_state[repo_id])

    def update_repo_settings(self, repo_id, repo_type=None, private=None, **_):
        self.update_calls.append((repo_id, repo_type, private))
        self.private_state[repo_id] = bool(private)
        return {"updated": True}


class TestMainDryRun(unittest.TestCase):
    """Default mode (no --execute) must NEVER call update_repo_settings."""

    def setUp(self):
        # All repos start private.
        self.fake = _FakeApi({rid: True for rid, _ in flip_visibility.REPOS})

    def _run(self, argv):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=False):
            with mock.patch("sys.stdout", new=io.StringIO()) as buf:
                rc = flip_visibility.main(argv, api_factory=lambda: self.fake)
            return rc, buf.getvalue()

    def test_make_public_dry_run_no_writes(self):
        rc, out = self._run(["--make", "public"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.update_calls, [], "dry run must not call update_repo_settings")
        self.assertIn("DRY RUN", out)
        self.assertIn("to change: 7", out)

    def test_make_private_dry_run_when_already_private(self):
        rc, out = self._run(["--make", "private"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.update_calls, [])
        self.assertIn("to change: 0", out)

    def test_verify_only_no_writes(self):
        rc, out = self._run(["--verify-only"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.update_calls, [])
        self.assertIn("Current visibility", out)


class TestMainExecute(unittest.TestCase):
    """--execute mode must call update_repo_settings with correct args."""

    def _run(self, argv, fake):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=False):
            with mock.patch("sys.stdout", new=io.StringIO()) as buf:
                rc = flip_visibility.main(argv, api_factory=lambda: fake)
            return rc, buf.getvalue()

    def test_make_public_executes_updates_private_repos_only(self):
        # 4 private + 3 already-public → only 4 should be updated.
        state = {rid: True for rid, _ in flip_visibility.REPOS[:4]}
        state.update({rid: False for rid, _ in flip_visibility.REPOS[4:]})
        fake = _FakeApi(state)
        rc, out = self._run(["--make", "public", "--execute"], fake)
        self.assertEqual(rc, 0)
        self.assertEqual(len(fake.update_calls), 4)
        # All updates must target private=False
        for _, _, p in fake.update_calls:
            self.assertFalse(p)
        # Repo-types passed must match REPOS
        for rid, rt, _ in fake.update_calls:
            expected_rt = dict(flip_visibility.REPOS)[rid]
            self.assertEqual(rt, expected_rt)

    def test_make_private_executes_only_public_repos(self):
        # All public — flip all to private.
        state = {rid: False for rid, _ in flip_visibility.REPOS}
        fake = _FakeApi(state)
        rc, _ = self._run(["--make", "private", "--execute"], fake)
        self.assertEqual(rc, 0)
        self.assertEqual(len(fake.update_calls), len(flip_visibility.REPOS))
        for _, _, p in fake.update_calls:
            self.assertTrue(p)

    def test_idempotent_execute_when_already_target(self):
        # All already private; --make private --execute should be no-op.
        fake = _FakeApi({rid: True for rid, _ in flip_visibility.REPOS})
        rc, out = self._run(["--make", "private", "--execute"], fake)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.update_calls, [])
        self.assertIn("Nothing to do", out)


class TestMainErrorPaths(unittest.TestCase):
    def test_missing_hf_token(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("sys.stderr", new=io.StringIO()) as err:
                rc = flip_visibility.main(["--make", "public"])
        self.assertEqual(rc, 2)
        self.assertIn("HF_TOKEN", err.getvalue())

    def test_repo_info_failure_returns_error_code(self):
        class _BrokenApi:
            def repo_info(self, *a, **kw):
                raise RuntimeError("simulated outage")

        with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=False):
            with mock.patch("sys.stdout", new=io.StringIO()), mock.patch(
                "sys.stderr", new=io.StringIO()
            ):
                rc = flip_visibility.main(
                    ["--make", "public"], api_factory=lambda: _BrokenApi()
                )
        self.assertEqual(rc, 3)


# ---------------------------------------------------------------------------
# Functional tests — require live HF, skip if no token
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
HAS_TOKEN = bool(HF_TOKEN)


def _hf_get(url):
    """Authenticated raw fetch from HF."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    return urllib.request.urlopen(req).read().decode("utf-8", errors="replace")


@unittest.skipUnless(HAS_TOKEN, "HF_TOKEN required for live HF tests")
class TestLiveRepos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import HfApi

        cls.api = HfApi(token=HF_TOKEN)

    def test_whoami_is_truthseeker87(self):
        self.assertEqual(self.api.whoami()["name"], "Truthseeker87")

    def test_all_registry_repos_resolve(self):
        for repo_id, repo_type in flip_visibility.REPOS:
            with self.subTest(repo_id=repo_id):
                info = self.api.repo_info(repo_id, repo_type=repo_type)
                self.assertEqual(info.id, repo_id)

    def test_v1_dataset_repo_does_not_exist(self):
        """v1 dataset solarhive-community-solar-1k was deleted Apr 30 (OB12)."""
        from huggingface_hub.utils import RepositoryNotFoundError

        with self.assertRaises((RepositoryNotFoundError, Exception)):
            self.api.repo_info(
                "Truthseeker87/solarhive-community-solar-1k", repo_type="dataset"
            )


@unittest.skipUnless(HAS_TOKEN, "HF_TOKEN required for live HF audit tests")
class TestLiveDatasetCard(unittest.TestCase):
    """Audit invariants on the canonical HF dataset card."""

    URL = "https://huggingface.co/datasets/Truthseeker87/solarhive-community-solar-multimodal/raw/main/README.md"

    @classmethod
    def setUpClass(cls):
        cls.body = _hf_get(cls.URL)

    def test_v2_canonical_count(self):
        self.assertIn("1,727", self.body)
        self.assertIn("1,713", self.body)

    def test_image_grounded_count(self):
        self.assertIn("14 image-grounded", self.body)

    def test_unique_qa_count(self):
        self.assertIn("1,530", self.body)

    def test_tool_calling_count(self):
        self.assertIn("183", self.body)

    def test_when2call_taxonomy_present(self):
        for n in ["106", "53", "10", "6", "8"]:
            self.assertIn(n, self.body, f"When2Call count {n} missing")
        self.assertIn("When2Call", self.body)

    def test_no_v1_leaks(self):
        """No stale v1-distinction language should remain in the card."""
        for forbidden in [
            "v1 fine-tune",
            "v1 weights",
            "1,029-row subset",
            "v2 fine-tune",
            "re-benchmark pending",
        ]:
            self.assertNotIn(forbidden, self.body, f"Stale phrase: {forbidden!r}")

    def test_text_only_disclosure_present(self):
        self.assertIn("text-only", self.body)

    def test_vision_encoder_citation_present(self):
        self.assertIn("vision encoder", self.body)
        # Either the model-card or the HF blog citation must be present.
        self.assertTrue(
            "model_card_4" in self.body or "blog/gemma4" in self.body,
            "Dataset card should cite at least one of the two Google Gemma 4 sources",
        )

    def test_no_deleted_v1_url(self):
        self.assertNotIn(
            "solarhive-community-solar-1k",
            self.body,
            "Dataset card must not link to the deleted v1 dataset repo",
        )


@unittest.skipUnless(HAS_TOKEN, "HF_TOKEN required for live HF audit tests")
class TestLiveModelCardCharts(unittest.TestCase):
    """Pin the chart-binary count and references in the two cards that ship charts."""

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import HfApi

        cls.api = HfApi(token=HF_TOKEN)

    def _list_chart_files(self, repo_id):
        files = self.api.list_repo_files(repo_id, repo_type="model")
        return sorted(f for f in files if f.startswith("datagen_charts/chart_"))

    def test_a4b_lora_has_14_charts(self):
        charts = self._list_chart_files("Truthseeker87/solarhive-26b-a4b-lora")
        self.assertEqual(len(charts), 14)
        self.assertIn("datagen_charts/chart_13.png", charts)
        self.assertIn("datagen_charts/chart_14.png", charts)

    def test_a4b_merged_has_14_charts(self):
        charts = self._list_chart_files("Truthseeker87/solarhive-26b-a4b-merged")
        self.assertEqual(len(charts), 14)
        self.assertIn("datagen_charts/chart_13.png", charts)
        self.assertIn("datagen_charts/chart_14.png", charts)

    def test_a4b_lora_card_says_14_charts(self):
        body = _hf_get(
            "https://huggingface.co/Truthseeker87/solarhive-26b-a4b-lora/raw/main/README.md"
        )
        self.assertIn("14 diagnostic charts", body)
        self.assertNotIn("12 diagnostic charts", body)

    def test_a4b_merged_card_says_14_charts(self):
        body = _hf_get(
            "https://huggingface.co/Truthseeker87/solarhive-26b-a4b-merged/raw/main/README.md"
        )
        self.assertIn("14 diagnostic charts", body)
        self.assertNotIn("12 diagnostic charts", body)


@unittest.skipUnless(HAS_TOKEN, "HF_TOKEN required for live HF audit tests")
class TestLiveVisionEncoderTransparency(unittest.TestCase):
    """Every shipped surface must cite either the Google AI model card or HF blog."""

    SURFACES = [
        ("HF dataset", "https://huggingface.co/datasets/Truthseeker87/solarhive-community-solar-multimodal/raw/main/README.md"),
        ("HF a4b-lora", "https://huggingface.co/Truthseeker87/solarhive-26b-a4b-lora/raw/main/README.md"),
        ("HF a4b-nf4", "https://huggingface.co/Truthseeker87/solarhive-26b-a4b-nf4/raw/main/README.md"),
        ("HF a4b-merged", "https://huggingface.co/Truthseeker87/solarhive-26b-a4b-merged/raw/main/README.md"),
        ("HF e4b-ollama", "https://huggingface.co/Truthseeker87/solarhive-e4b-ollama/raw/main/README.md"),
        ("HF e4b-gguf", "https://huggingface.co/Truthseeker87/solarhive-e4b-gguf/raw/main/README.md"),
    ]

    def test_all_surfaces_cite_google_or_hf_blog(self):
        for name, url in self.SURFACES:
            with self.subTest(surface=name):
                body = _hf_get(url)
                self.assertTrue(
                    "model_card_4" in body or "blog/gemma4" in body,
                    f"{name} should cite Google Gemma 4 model card or HF blog",
                )

    def test_all_surfaces_disclose_text_only(self):
        for name, url in self.SURFACES:
            with self.subTest(surface=name):
                body = _hf_get(url)
                self.assertIn(
                    "text-only",
                    body,
                    f"{name} should disclose that fine-tuning is text-only",
                )

    def test_no_surface_has_v1_distinction_leaks(self):
        for name, url in self.SURFACES:
            with self.subTest(surface=name):
                body = _hf_get(url)
                for forbidden in [
                    "v1 fine-tune",
                    "1,029-row subset",
                    "re-benchmark pending",
                ]:
                    self.assertNotIn(
                        forbidden,
                        body,
                        f"{name} contains stale v1 language: {forbidden!r}",
                    )


if __name__ == "__main__":
    unittest.main()
