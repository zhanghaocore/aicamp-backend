import argparse
import logging

from flask import Flask
from utils.timer import PerformanceTimer

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

_OP = {
        'model': {'missing_wedge_angle': 30, 'SNR': 0.05},
        'ctf': {'pix_size': 1.0, 'Dz': -5.0, 'voltage': 300, 'Cs': 2.0, 'sigma': 0.4}
    }
_LOC_PROPORTION = 0.1


@app.route("/")
def home():
    with PerformanceTimer(__name__):
        return {"message": "MBZUAI CIAI AI Boot Camp Service"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    if args.debug:
        log_model = logging.DEBUG
    else:
        log_model = logging.INFO
    logging.basicConfig(level=log_model)

    app.run(debug=args.debug, host=args.host, port=args.port)
