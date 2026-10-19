"""fgPROMS launcher: starts the local server and opens the browser."""
import socket
import threading
import time
import webbrowser

import uvicorn

from backend.app import app


def find_port():
    for p in (8765, 8766, 8767, 8768, 8769):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return 8765


def main():
    port = find_port()

    def open_browser():
        for _ in range(80):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.25)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    print("=" * 52)
    print("  fgPROMS is running at:  http://127.0.0.1:%d" % port)
    print("  Press Ctrl+C in this window to stop the server.")
    print("=" * 52)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
