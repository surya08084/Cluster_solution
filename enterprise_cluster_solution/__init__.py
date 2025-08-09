from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import socket
import threading
import webbrowser


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@dataclass
class ClusteringSolution:
    host: str = "127.0.0.1"
    port: Optional[int] = None

    def launch_ui(self, open_browser: bool = True) -> str:
        try:
            # Lazy import heavy deps
            import uvicorn  # type: ignore
            from .server import build_app  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import server dependencies. Ensure package dependencies are installed."
            ) from exc

        selected_port = self.port or _find_free_port(self.host)
        app = build_app()

        config = uvicorn.Config(app, host=self.host, port=selected_port, log_level="info")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        url = f"http://{self.host}:{selected_port}"
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url