import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from algorithms.genetic import GeneticAlgorithm
from algorithms.local_search import LocalSearch

HOST = "localhost"
PORT = 8000


class RequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path: str = urlparse(self.path).path

        if path == "/api/health":
            self.send_json({"status": "ok"})
        elif path == "/api/local-search":
            self.send_json(LocalSearch().solve())
        elif path == "/api/genetic":
            self.send_json(GeneticAlgorithm().solve())
        else:
            self.send_json({"error": "Ruta no encontrada"}, status=404)

    def send_json(self, data: dict, status: int = 200):
        body: bytes = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    server: HTTPServer = HTTPServer((HOST, PORT), RequestHandler)

    print(f"Backend de focos escuchando en http://{HOST}:{PORT}")
    print("Endpoints:")
    print("  GET /api/health")
    print("  GET /api/local-search")
    print("  GET /api/genetic")

    server.serve_forever()


if __name__ == "__main__":
    main()
