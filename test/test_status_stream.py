import threading
import unittest

from status_stream import StatusStream


class StatusStreamTests(unittest.TestCase):
    def test_callback_can_read_snapshot_without_deadlock(self):
        stream = StatusStream()
        seen = []

        def callback(_line):
            seen.extend(stream.snapshot())

        stream.register(callback)

        thread = threading.Thread(target=stream.emit, args=("agent", "Received task: hi"))
        thread.start()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(seen, ["agent Received task: hi"])


if __name__ == "__main__":
    unittest.main()
