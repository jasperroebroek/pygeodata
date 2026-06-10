import webbrowser
from wsgiref.simple_server import make_server

from pygeodata.registry_browser.web import create_app


def open_registry_browser(port: int = 0) -> None:
    """Start the registry browser and open it in the default browser."""
    app = create_app()
    server = make_server('127.0.0.1', port, app)
    host, actual_port = server.server_address
    url = f'http://{host}:{actual_port}'
    print(url)
    webbrowser.open(url)
    server.serve_forever()
