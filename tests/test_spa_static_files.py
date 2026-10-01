"""Security tests for the SPA (React frontend) static file routes.

Covers the catch-all route registered by ``register_spa_routes``:
- legitimate frontend files and client-side routes are still served
- path traversal (absolute paths, ``..`` chains, encoded dot-dot segments,
  symlinks pointing outside the root, NUL bytes) never leaks files outside
  the static root
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import register_spa_routes, resolve_static_path

SECRET = "TOP-SECRET-OPENAI-KEY-sk-should-never-leak"
INDEX_HTML = "<!doctype html><html><body>SPA-INDEX</body></html>"


@pytest.fixture
def site(tmp_path: Path) -> SimpleNamespace:
    """Build a frontend build directory plus files/symlinks outside of it."""
    root = tmp_path / "frontend"
    (root / "static" / "js").mkdir(parents=True)
    (root / "index.html").write_text(INDEX_HTML)
    (root / "favicon.ico").write_bytes(b"\x00\x01FAVICON")
    (root / "static" / "js" / "main.js").write_text("console.log('main');")

    # Secret file outside the static root
    secret = tmp_path / "secret.txt"
    secret.write_text(SECRET)

    # Sibling directory sharing the root's name as a prefix ("frontend-evil")
    sibling = tmp_path / "frontend-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text(SECRET)

    # Symlinks inside the root pointing outside of it
    (root / "leak.txt").symlink_to(secret)
    (root / "rel_leak.txt").symlink_to("../secret.txt")
    (root / "static" / "js" / "leak.js").symlink_to("../../../secret.txt")
    (root / "linked_dir").symlink_to(tmp_path, target_is_directory=True)
    # Symlink inside the root pointing inside of it (legitimate)
    (root / "alias.js").symlink_to("static/js/main.js")
    # Symlink loop inside the root
    (root / "loop").symlink_to(root / "loop")

    return SimpleNamespace(root=root, secret=secret)


@pytest.fixture
def spa_app(site: SimpleNamespace) -> FastAPI:
    """Fresh FastAPI app with an API route and the SPA routes registered."""
    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    register_spa_routes(app, site.root)
    return app


@pytest.fixture
def client(spa_app: FastAPI) -> TestClient:
    """Test client for the SPA app."""
    return TestClient(spa_app)


async def _raw_get(app: FastAPI, path: str) -> tuple[int, bytes]:
    """Send a GET with an unnormalised path straight to the ASGI app.

    httpx (and therefore TestClient) collapses literal ``../`` segments, while
    uvicorn passes them through (e.g. ``curl --path-as-is``), so literal
    dot-dot chains are exercised by calling the app directly.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    messages = [{"type": "http.request", "body": b"", "more_body": False}]
    response: dict = {"status": None, "body": b""}

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return response["status"], response["body"]


def _render(template: str, site: SimpleNamespace) -> str:
    """Fill a payload template with the (tmp_path dependent) secret location.

    Placeholders: ``{secret}`` absolute path, ``{fs_secret}`` the same path
    without the leading slash, ``{fs_secret_enc}`` with ``/`` encoded as ``%2f``.
    """
    fs_secret = site.secret.as_posix().lstrip("/")
    return template.format(
        secret=site.secret.as_posix(),
        fs_secret=fs_secret,
        fs_secret_enc=fs_secret.replace("/", "%2f"),
    )


DOTDOT_ENC_SLASH = "..%2f" * 20
DOTDOT_ENC_DOTS = "%2e%2e/" * 20
DOTDOT_LITERAL = "../" * 20


class TestLegitimateRequests:
    """Normal frontend requests keep working."""

    def test_root_serves_index(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == INDEX_HTML

    def test_root_level_file_is_served(self, client):
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.content == b"\x00\x01FAVICON"

    def test_static_asset_is_served(self, client):
        response = client.get("/static/js/main.js")
        assert response.status_code == 200
        assert response.text == "console.log('main');"

    def test_symlink_inside_root_is_served(self, client):
        response = client.get("/alias.js")
        assert response.status_code == 200
        assert response.text == "console.log('main');"

    @pytest.mark.parametrize("path", ["/chat/123", "/documents", "/settings/profile/edit"])
    def test_unknown_client_route_falls_back_to_index(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert response.text == INDEX_HTML

    def test_api_routes_are_not_shadowed(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_unknown_api_route_returns_404(self, client):
        response = client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert INDEX_HTML not in response.text

    def test_missing_index_returns_404(self, tmp_path):
        root = tmp_path / "empty_build"
        root.mkdir()
        app = FastAPI()
        register_spa_routes(app, root)

        response = TestClient(app).get("/chat/123")
        assert response.status_code == 404


class TestPathTraversalViaHttp:
    """Traversal payloads sent through a regular HTTP client."""

    @pytest.mark.parametrize(
        "url_template",
        [
            # Absolute path: os.path.join() would discard the static root
            pytest.param("http://testserver/{secret}", id="double-slash"),
            pytest.param("/..%2fsecret.txt", id="encoded-slash"),
            pytest.param("/" + DOTDOT_ENC_SLASH + "{fs_secret_enc}", id="encoded-slash-chain"),
            pytest.param("/%2e%2e/secret.txt", id="encoded-dots"),
            pytest.param("/%2E%2E%2Fsecret.txt", id="encoded-dots-and-slash"),
            pytest.param("/" + DOTDOT_ENC_DOTS + "{fs_secret}", id="encoded-dots-chain"),
            pytest.param("/chat/%2e%2e/%2e%2e/secret.txt", id="nested-encoded-dots"),
            pytest.param("/..%2ffrontend-evil%2fsecret.txt", id="sibling-prefix-dir"),
            pytest.param("/leak.txt", id="symlinked-file"),
            pytest.param("/rel_leak.txt", id="relative-symlinked-file"),
            pytest.param("/static/js/leak.js", id="symlinked-static-asset"),
            pytest.param("/linked_dir/secret.txt", id="symlinked-dir"),
            # Before Python 3.13 resolve() stops at a symlink loop and collapses
            # the following ".." lexically, leaving the outward symlink unresolved
            pytest.param("/loop/%2e%2e/leak.txt", id="loop-then-symlinked-file"),
            pytest.param("/loop/..%2fleak.txt", id="loop-then-encoded-slash"),
            pytest.param("/loop/%2e%2e/rel_leak.txt", id="loop-then-relative-symlink"),
            pytest.param("/loop/%2e%2e/static/js/leak.js", id="loop-then-nested-symlink"),
            pytest.param("/loop/%2e%2e/linked_dir/secret.txt", id="loop-then-symlinked-dir"),
            pytest.param("/loop/x/%2e%2e/%2e%2e/leak.txt", id="loop-subpath-then-symlink"),
            pytest.param("/..%2fsecret.txt%00", id="nul-byte-traversal"),
            pytest.param("/chat%00", id="nul-byte"),
        ],
    )
    def test_traversal_is_rejected(self, client, site, url_template):
        response = client.get(_render(url_template, site))

        assert response.status_code == 404
        assert SECRET not in response.text

    def test_symlink_loop_does_not_crash(self, client):
        response = client.get("/loop")

        assert response.status_code in (200, 404)
        assert SECRET not in response.text


class TestPathTraversalRawAsgi:
    """Literal ``../`` chains as uvicorn delivers them (curl --path-as-is)."""

    @pytest.mark.parametrize(
        "path_template",
        [
            pytest.param("/../secret.txt", id="single-dotdot"),
            pytest.param("/" + DOTDOT_LITERAL + "{fs_secret}", id="chain"),
            pytest.param("/chat/../../secret.txt", id="nested-dotdot"),
            pytest.param("/{secret}", id="double-slash"),
            pytest.param("/../frontend-evil/secret.txt", id="sibling-prefix-dir"),
            pytest.param("/loop/../leak.txt", id="loop-then-symlinked-file"),
            pytest.param("/loop/../linked_dir/secret.txt", id="loop-then-symlinked-dir"),
        ],
    )
    async def test_literal_dotdot_is_rejected(self, spa_app, site, path_template):
        status_code, body = await _raw_get(spa_app, _render(path_template, site))

        assert status_code == 404
        assert SECRET.encode() not in body

    async def test_dotdot_resolving_inside_root_is_allowed(self, spa_app):
        status_code, body = await _raw_get(spa_app, "/chat/../favicon.ico")

        assert status_code == 200
        assert body == b"\x00\x01FAVICON"


class TestResolveStaticPath:
    """Unit tests for the containment helper using raw (undecoded) strings."""

    @pytest.fixture
    def root(self, site):
        return site.root.resolve()

    @pytest.mark.parametrize(
        "request_path",
        [
            "../x",
            "..",
            "/etc/passwd",
            "//etc/passwd",
            "a/../../x",
            "../../../../../../../../etc/passwd",
            "../frontend-evil/secret.txt",
            "leak.txt",
            "rel_leak.txt",
            "static/js/leak.js",
            "linked_dir/secret.txt",
            # Symlink loop followed by ".." (unresolved outward symlink before 3.13)
            "loop/../leak.txt",
            "loop/../rel_leak.txt",
            "loop/../static/js/leak.js",
            "loop/../linked_dir/secret.txt",
            "loop/x/../../leak.txt",
            "a\x00b",
            "../secret.txt\x00.html",
        ],
    )
    def test_escaping_or_invalid_paths_return_none(self, root, request_path):
        assert resolve_static_path(root, request_path) is None

    def test_absolute_secret_path_returns_none(self, root, site):
        assert resolve_static_path(root, os.fspath(site.secret)) is None
        assert resolve_static_path(root, "/" + os.fspath(site.secret)) is None

    @pytest.mark.parametrize(
        ("request_path", "expected"),
        [
            ("index.html", "index.html"),
            ("favicon.ico", "favicon.ico"),
            ("static/js/main.js", "static/js/main.js"),
            ("chat/../favicon.ico", "favicon.ico"),
            ("chat/123", "chat/123"),
            ("alias.js", "static/js/main.js"),
        ],
    )
    def test_contained_paths_are_resolved(self, root, request_path, expected):
        assert resolve_static_path(root, request_path) == root / expected

    def test_empty_path_resolves_to_root(self, root):
        assert resolve_static_path(root, "") == root
