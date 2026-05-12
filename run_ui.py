import http.server
import socketserver
import os
import socket

DIRECTORY = "ui"

def get_free_port(start_port=3000):
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            port += 1

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    
    port = get_free_port()
        
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"UI Server running at http://localhost:{port}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
