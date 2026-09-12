"""Server entry point."""

import argparse
import logging
import threading

from .app import create_app
from .config import get_config
from .webui import create_webui_app

logger = logging.getLogger(__name__)


def main():
    """Run the image generation server."""
    parser = argparse.ArgumentParser(description="Image Generation Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--webui-host", default="0.0.0.0", help="WebUI host to bind to")
    parser.add_argument("--webui-port", type=int, default=5001, help="WebUI port to bind to")
    parser.add_argument("--no-webui", action="store_true", help="Disable webui")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    config = get_config()
    host = args.host or config.host
    port = args.port or config.port
    
    app = create_app()
    
    if not args.no_webui:
        webui_app = create_webui_app()
        webui_thread = threading.Thread(
            target=lambda: webui_app.run(
                host=args.webui_host,
                port=args.webui_port,
                debug=False,
                use_reloader=False
            ),
            daemon=True,
            name="WebUIThread"
        )
        webui_thread.start()
        logger.info(f"Starting webui on {args.webui_host}:{args.webui_port}")
    
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
