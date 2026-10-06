import json
import unittest
from clinical_review.api import handle
from clinical_review.service import Service
from clinical_review.store import Store


class 基线行为测试(unittest.TestCase):
    def test_health(self):
        result = json.loads(handle(json.dumps({"action": "health"}), Service(Store())))
        self.assertEqual(result["status"], "ok")

    def test_create_and_get_session(self):
        service = Service(Store())
        created = service.create_session("s-1", "脱敏后的提问", ["normal"])
        self.assertEqual(created["state"], "open")
        self.assertEqual(service.get_session("s-1")["risk_labels"], ("normal",))


if __name__ == "__main__":
    unittest.main()
