#!./env-parliament/bin/python3
"""
For serving local files on a web browser.

Simply run:

`python -m tools.simple_http_server`

and open files relative to the location of this simple server.
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler, test


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleHTTPRequestHandler.end_headers(self)


if __name__ == "__main__":
    test(
        CORSRequestHandler,
        HTTPServer,
        port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000,
    )
