"""CPU-only regression tests for website assembly and legacy URL handling."""
import tempfile
import unittest
from pathlib import Path

from build_site import LEGACY_ROUTES, assemble
from check_site import check


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sphinx = self.root / "html"
        self.landing = self.root / "landing"
        self.output = self.root / "site"
        self.sphinx.mkdir()
        self.landing.mkdir()
        for name in {*LEGACY_ROUTES.values(), "index.html", "guide/topic.html"}:
            path = self.sphinx / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<html><body id="section">Documentation</body></html>')
        (self.sphinx / "_static").mkdir()
        (self.sphinx / "_static/logo.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        (self.landing / "index.html").write_text(
            '<html><head><link href="landing/style.css" rel="stylesheet"></head>'
            '<body id="main"><a href="#main">Home</a>'
            '<a href="docs/guide/topic.html#section">Docs</a></body></html>')
        (self.landing / "style.css").write_text("body { color: black; }")

    def assemble(self):
        assemble(self.sphinx, self.landing, self.output)

    def test_assembly_and_local_links(self):
        self.assemble()
        check(self.output)
        self.assertEqual((self.output / "index.html").read_text(),
                         (self.landing / "index.html").read_text())
        self.assertTrue((self.output / "_static/logo.svg").is_file())
        self.assertTrue((self.output / "docs/_static/logo.svg").is_file())

    def test_redirects_keep_prefix_query_and_fragment(self):
        self.assemble()
        text = (self.output / "guide/topic.html").read_text()
        self.assertIn('location.replace("../docs/guide/topic.html"', text)
        self.assertIn("location.search + location.hash", text)
        old = (self.output / "user_guide/landing_page.html").read_text()
        self.assertIn("../docs/index.html", old)

    def test_refuses_existing_output_without_modification(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("user work")
        with self.assertRaisesRegex(ValueError, "Output exists"):
            self.assemble()
        self.assertEqual(sentinel.read_text(), "user work")

    def test_refuses_overlapping_output(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            assemble(self.sphinx, self.landing, self.sphinx / "site")

    def test_missing_legacy_destination_rejected(self):
        (self.sphinx / "how_do_i.html").unlink()
        with self.assertRaisesRegex(ValueError, "Missing legacy route"):
            self.assemble()

    def test_missing_asset_detected(self):
        self.assemble()
        (self.output / "landing/style.css").unlink()
        with self.assertRaisesRegex(SystemExit, "missing local target"):
            check(self.output)

    def test_invalid_homepage_fragment_detected(self):
        self.assemble()
        (self.output / "index.html").write_text('<a href="docs/index.html#absent">Bad</a>')
        with self.assertRaisesRegex(SystemExit, "missing anchor"):
            check(self.output)

    def test_wrong_redirect_detected(self):
        self.assemble()
        (self.output / "guide/topic.html").write_text('<a href="../docs/index.html">Wrong</a>')
        with self.assertRaisesRegex(SystemExit, "Wrong legacy redirect"):
            check(self.output)

    def test_invalid_documentation_fragment_detected(self):
        self.assemble()
        (self.output / "docs/index.html").write_text(
            '<a href="guide/topic.html#absent">Bad documentation anchor</a>')
        with self.assertRaisesRegex(SystemExit, "missing anchor"):
            check(self.output)

    def test_project_home_cannot_link_to_docs_itself(self):
        self.assemble()
        (self.output / "docs/index.html").write_text('<a data-grid-home href="#">Project home</a>')
        with self.assertRaisesRegex(SystemExit, "Project home must return to cover"):
            check(self.output)


if __name__ == "__main__":
    unittest.main()
