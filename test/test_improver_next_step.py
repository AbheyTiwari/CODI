import unittest

from core.improver import Improver
from state.temp_db import RunState


class ImproverNextStepTests(unittest.TestCase):
    def test_next_step_uses_plan_index_without_llm(self):
        state = RunState(iteration=2)
        state.plan = "Create files"
        state.plan_steps = ["Write Book.java", "Write Member.java", "Write Loan.java"]

        improver = Improver.__new__(Improver)
        improver._call = lambda *args, **kwargs: self.fail("next_step called the LLM")

        decision = improver.next_step(state)

        self.assertEqual(decision, {"step": "Write Member.java", "done": False})
        self.assertEqual(state.llm_exchanges, [])

    def test_next_step_falls_back_to_llm_for_correction(self):
        state = RunState(iteration=2)
        state.plan = "Create files\n[CORRECTION]: fix compile error"
        state.plan_steps = ["Write Book.java", "Write Member.java", "Write Loan.java"]

        improver = Improver.__new__(Improver)
        improver._call = lambda *args, **kwargs: '{"step":"Fix compile error","done":false}'

        decision = improver.next_step(state)

        self.assertEqual(decision, {"step": "Fix compile error", "done": False})
        self.assertEqual(state.llm_exchanges[0]["role"], "improver_next_step")


if __name__ == "__main__":
    unittest.main()
