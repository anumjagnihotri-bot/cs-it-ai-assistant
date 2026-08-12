import unittest
from types import SimpleNamespace

import app


class TestExtractReply(unittest.TestCase):
    def test_extracts_direct_text(self):
        response = SimpleNamespace(text="hello from gemini")
        self.assertEqual(app.extract_reply(response), "hello from gemini")

    def test_extracts_nested_candidate_text(self):
        response = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="nested answer")]
                    )
                )
            ],
        )
        self.assertEqual(app.extract_reply(response), "nested answer")

    def test_returns_empty_for_blocked_or_empty_response(self):
        response = SimpleNamespace(text="", candidates=[])
        self.assertEqual(app.extract_reply(response), "")


if __name__ == "__main__":
    unittest.main()
