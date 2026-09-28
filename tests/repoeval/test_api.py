import unittest

from missy.repoeval.api import FoundryAPI
from missy.repoeval.mcp import FoundryMCP, MCPError


class Principal:
    subject = "operator"
    project_id = "p1"
    permissions = frozenset({"read", "execute", "cancel"})


class Service:
    def __init__(self):
        self.calls = []

    def list_repositories(self, p):
        self.calls.append(("repos", p))
        return ["r1"]

    def benchmark_plan(self, p, w):
        self.calls.append(("plan", p, w))
        return {"id": "plan-x"}

    def benchmark_start(self, p, plan, key):
        self.calls.append(("start", plan, key))
        return {"id": "run-x"}

    def run_status(self, p, run):
        return {"id": run, "state": "submitted"}

    def run_cancel(self, p, run):
        self.calls.append(("cancel", run))
        return {"id": run, "state": "cancelled"}

    def snapshot_start(self, p, repo, sha, key):
        self.calls.append(("snapshot", repo, sha, key))
        return {"id": "snap-x"}

    def snapshot_status(self, p, snap):
        return {"id": snap}

    def compare_runs(self, p, ids):
        self.calls.append(("compare", p.project_id, ids))
        return {
            "project_id": p.project_id,
            "comparison_id": "comparison-" + "a" * 24,
            "run_ids": ids,
            "comparable": True,
            "pairs": [],
        }

    def artifacts(self, p, resource):
        self.calls.append(("artifacts", p.project_id, resource))
        return {"project_id": p.project_id, "run_id": resource, "artifacts": []}

    def report_draft(self, p, ids):
        self.calls.append(("report", p.project_id, ids))
        return {
            "project_id": p.project_id,
            "report_id": "report-" + "b" * 24,
            "status": "draft",
            "published": False,
            "run_ids": ids,
            "groups": [],
        }


class APITests(unittest.TestCase):
    def setUp(self):
        self.service = Service()
        self.api = FoundryAPI(
            self.service, lambda h: Principal() if h.get("Authorization") == "Bearer test" else None
        )
        self.headers = {"Authorization": "Bearer test"}

    def test_auth_required(self):
        r = self.api.handle("GET", "/v1/projects/p1/repositories", {}, {})
        self.assertEqual(r.status, 401)
        self.assertEqual(r.body["error"]["category"], "unauthenticated")

    def test_project_scope(self):
        r = self.api.handle("GET", "/v1/projects/other/repositories", self.headers, {})
        self.assertEqual(r.status, 403)

    def test_comparison_artifact_and_draft_report_scoped_wires(self):
        ids = ["run-alpha0001", "run-beta00002"]
        for method, path, body, action in (
            ("POST", "/api/projects/p1/compare", {"run_ids": ids}, "compare"),
            ("GET", "/api/projects/p1/runs/run-alpha0001/artifacts", {}, "artifacts"),
            ("POST", "/api/projects/p1/report", {"run_ids": ids}, "report"),
        ):
            with self.subTest(action=action):
                r = self.api.handle(method, path, self.headers, body)
                self.assertEqual(r.status, 200)
                self.assertEqual(r.body["data"]["project_id"], "p1")
                self.assertIn(action, [c[0] for c in self.service.calls])
                count = len(self.service.calls)
                foreign = self.api.handle(
                    method, path.replace("/p1/", "/other/"), self.headers, body
                )
                self.assertEqual(foreign.status, 403)
                self.assertEqual(len(self.service.calls), count)
        self.assertFalse(
            self.api.handle("POST", "/api/projects/p1/report", self.headers, {"run_ids": ids}).body[
                "data"
            ]["published"]
        )

    def test_read_routes_refuse_invalid_ids_before_dispatch(self):
        for method, path, body in (
            ("POST", "/api/projects/p1/compare", {"run_ids": ["run-a", "run-a"]}),
            ("POST", "/api/projects/p1/report", {"run_ids": ["../foreign"]}),
            ("POST", "/api/projects/p1/report", {"run_ids": ["run-a"], "publish": True}),
            ("GET", "/api/projects/p1/runs/%2Fforeign/artifacts", {}),
        ):
            before = len(self.service.calls)
            self.assertEqual(self.api.handle(method, path, self.headers, body).status, 400)
            self.assertEqual(len(self.service.calls), before)

    def test_idempotency_required_then_forwarded(self):
        r = self.api.handle(
            "POST", "/v1/projects/p1/benchmark/start", self.headers, {"plan_id": "plan-x"}
        )
        self.assertEqual(r.status, 400)
        r = self.api.handle(
            "POST",
            "/v1/projects/p1/benchmark/start",
            {**self.headers, "Idempotency-Key": "idem-0001"},
            {"plan_id": "plan-x"},
        )
        self.assertEqual(r.status, 202)
        self.assertIn(("start", "plan-x", "idem-0001"), self.service.calls)

    def test_unknown_route_refused(self):
        r = self.api.handle("POST", "/v1/projects/p1/delete", self.headers, {})
        self.assertEqual(r.status, 404)

    def test_failure_is_stable_and_redacted(self):
        class Broken(Service):
            def list_repositories(self, p):
                raise RuntimeError("secret provider response")

        r = FoundryAPI(Broken(), lambda h: Principal()).handle(
            "GET", "/v1/projects/p1/repositories", {}, {}
        )
        self.assertEqual(r.status, 503)
        self.assertEqual(r.body["error"]["category"], "service_failure")
        self.assertNotIn("secret", r.body["error"]["message"])

    def test_mcp_refuses_unexposed_mutations_and_forwards_idempotency(self):
        mcp = FoundryMCP(self.service, Principal())
        with self.assertRaises(MCPError):
            mcp.call("repoeval_delete", {"run_id": "x"})
        result = mcp.call(
            "repoeval_benchmark_start", {"plan_id": "plan-x", "idempotency_key": "idem-0001"}
        )
        self.assertEqual(result["id"], "run-x")
        self.assertIn(("start", "plan-x", "idem-0001"), self.service.calls)

    def test_mcp_rejects_missing_arguments(self):
        with self.assertRaises(MCPError):
            FoundryMCP(self.service, Principal()).call("repoeval_benchmark_start", {"plan_id": "x"})

    def test_cancel_requires_idempotency_at_http_boundary(self):
        r = self.api.handle("POST", "/v1/projects/p1/runs/run-x/cancel", self.headers, {})
        self.assertEqual(r.status, 400)

    def test_cancel_validates_header_but_is_state_idempotent_not_keyed(self):
        headers = {**self.headers, "Idempotency-Key": "cancel-0001"}
        r = self.api.handle("POST", "/v1/projects/p1/runs/run-x/cancel", headers, {})
        self.assertEqual(r.status, 202)
        self.assertEqual(r.body["data"]["state"], "cancelled")
        self.assertIn(("cancel", "run-x"), self.service.calls)

    def test_categorized_backend_errors_never_expose_exception_text(self):
        secret = "Bearer abc123 api_key=supersecret"
        messages = {
            "authorization": "Operation is not permitted",
            "forbidden": "Operation is not permitted",
            "missing": "Resource not found",
            "not_found": "Resource not found",
            "invalid_input": "Request input is invalid",
            "quota": "Request exceeds an allowed limit",
            "policy": "Request is not allowed by policy",
            "conflict": "Request conflicts with the current resource state",
            "unavailable": "Required service capability is unavailable",
            "evidence": "Required evidence is unavailable or invalid",
        }
        for category, expected_message in messages.items():

            class BackendError(Exception):
                def __init__(self, bound_category=category):
                    self.category = bound_category
                    super().__init__(secret)

            class Broken(Service):
                def list_repositories(self, p):
                    raise BackendError()

            with self.subTest(category=category):
                r = FoundryAPI(Broken(), lambda h: Principal()).handle(
                    "GET", "/v1/projects/p1/repositories", {}, {}
                )
                self.assertEqual(r.body["error"]["message"], expected_message)
                self.assertNotIn(secret, repr(r.body))
                self.assertNotIn("abc123", repr(r.body))
                self.assertNotIn("supersecret", repr(r.body))

    def test_unsupported_publication_mutation_denied(self):
        m = FoundryMCP(self.service, Principal())
        with self.assertRaises(MCPError) as error:
            m.call("repoeval_publish_report", {"run_id": "run-x"})
        self.assertEqual(error.exception.category, "unsupported_operation")

    def test_mcp_backend_exception_text_and_category_are_not_exposed(self):
        secret = "secret-from-authorization-header"

        class BackendError(Exception):
            def __init__(self, category):
                self.category = category
                super().__init__(f"Bearer {secret}")

        for backend_category, expected in (
            ("authorization", "forbidden"),
            ("missing", "not_found"),
            ("invalid_input", "invalid_request"),
            ("conflict", "conflict"),
            (f"secret-{secret}", "service_failure"),
            ([secret], "service_failure"),
        ):

            class Broken(Service):
                def list_repositories(self, p, bound_category=backend_category):
                    raise BackendError(bound_category)

            with self.subTest(category=backend_category), self.assertRaises(MCPError) as error:
                FoundryMCP(Broken(), Principal()).call("repoeval_repositories", {})
            self.assertEqual(error.exception.category, expected)
            self.assertNotIn(secret, repr(error.exception))
            self.assertNotIn("Bearer", str(error.exception))

        class Crashed(Service):
            def list_repositories(self, p):
                raise RuntimeError(f"Bearer {secret}")

        with self.assertRaises(MCPError) as error:
            FoundryMCP(Crashed(), Principal()).call("repoeval_repositories", {})
        self.assertEqual(error.exception.category, "service_failure")
        self.assertEqual(str(error.exception), "Application service failed")

    def test_mcp_does_not_trust_backend_mcp_error_message(self):
        class Crashed(Service):
            def list_repositories(self, p):
                raise MCPError("invalid_request", "Bearer topsecret")

        with self.assertRaises(MCPError) as error:
            FoundryMCP(Crashed(), Principal()).call("repoeval_repositories", {})
        self.assertEqual(error.exception.category, "invalid_request")
        self.assertEqual(str(error.exception), "Request is invalid")


if __name__ == "__main__":
    unittest.main()
