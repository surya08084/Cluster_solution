from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import socket
import threading
import webbrowser


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@dataclass
class ClusteringSolution:
    host: Optional[str] = None
    port: Optional[int] = None

    def launch_ui(self, open_browser: bool = True) -> str:
        try:
            import uvicorn  # type: ignore
            from .server import build_app  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import server dependencies. Ensure package dependencies are installed."
            ) from exc

        # Resolve bind host/port from args or env
        bind_host = (self.host or os.getenv("ECS_HOST") or os.getenv("HOST") or "0.0.0.0").strip()
        bind_port = self.port or int(os.getenv("ECS_PORT") or os.getenv("PORT") or 0)
        if not bind_port:
            # find free port on loopback for probing, but bind on requested host
            bind_port = _find_free_port("127.0.0.1")

        app = build_app()

        config = uvicorn.Config(app, host=bind_host, port=bind_port, log_level="info")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Prefer a friendly URL: localhost if binding to 0.0.0.0
        display_host = "localhost" if bind_host == "0.0.0.0" else bind_host
        url = f"http://{display_host}:{bind_port}"
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url