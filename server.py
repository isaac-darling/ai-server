from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import socket, cgi, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

MODEL = tf.keras.models.load_model("test_model")

class PredictionServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
    def do_HEAD(self):
        self._set_headers()
    def do_GET(self):
        self._set_headers()
        self.wfile.write(b"<html><p>We apologize again for this webpage. Those responsible for sacking the people who have just been sacked have been sacked.</p></html>")
    def do_POST(self):
        self._set_headers()
        form = cgi.FieldStorage(
                fp = self.rfile,
                headers = self.headers,
                environ = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})

        file = form["payload"].value

        image = tf.image.rgb_to_grayscale(tf.io.decode_jpeg(file))
        image = tf.expand_dims(image, 0)
        normal_image = tf.cast(image, tf.float32) / 255.
        result = np.argmax(MODEL.predict(normal_image))

        self.wfile.write(f"{result = }".encode())

def primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "localhost"
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    ip = "localhost" #primary_ip()

    with HTTPServer((ip, 8001), PredictionServer) as server:
        try:
            print(f"http://{ip}:8001")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

#curl -F payload=@img.npz http://localhost:8001
