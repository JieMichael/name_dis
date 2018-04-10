#!/usr/bin/env python
"""
Very simple HTTP server in python.
Usage::
    ./dummy-web-server.py [<port>]
Send a GET request::
    curl http://localhost
Send a HEAD request::
    curl -I http://localhost
Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import pymysql
import os
import sys
import traceback
import _pickle as cpickle
from HttpDis import *
# import socketserver


class S(BaseHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(b"Please Use Post")

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        try:
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            print("length = ",content_length)
            sys.stdout.flush()
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself
            post_data = str(post_data,encoding = "utf-8")
            result = HttpDis(post_data)
            print ("len of result",len(result))
            print ("type of result",type(result))
            self._set_headers()
            self.wfile.write(result)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print(str(e))
            #self.wfile.write(b'Error: ' + str(e).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=S, port=10080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


if __name__ == "__main__":
    # load_ref()
    run()
