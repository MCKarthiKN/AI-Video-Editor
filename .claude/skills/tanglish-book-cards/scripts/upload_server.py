"""Tiny local server for rendering cards.

Serves SITE_DIR (a copy of QUOTE CARDS/index.html plus cards.txt) and saves
anything POSTed to /<name>.png into OUT_DIR. The browser job in
assets/render_job.js uploads every PNG here.

Usage (run in the background, from the scratchpad, not from the Desktop):
  python3 upload_server.py OUT_DIR SITE_DIR [PORT]     # default port 8766
Stop it afterwards with:  pkill -f "upload_server.py"
"""
import http.server, os, sys
OUT, SITE = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8766

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SITE, **k)
    def log_message(self, *a):
        pass
    def do_POST(self):
        name = os.path.basename(self.path.split("?")[0])
        n = int(self.headers["Content-Length"])
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(self.rfile.read(n))
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"ok")

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
