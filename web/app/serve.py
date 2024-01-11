import argparse
import logging
import time
import os


from flask import Flask, request, jsonify, abort, make_response
from werkzeug.exceptions import HTTPException, BadRequest
from utils.timer import PerformanceTimer

from fastchat_prompt import FastChatLLM

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)


@app.route("/")
def home():
    with PerformanceTimer(__name__):
        return {"message": "MBZUAI CIAI AI Boot Camp Service. v-0.0.1"}

@app.route("/api/chatbot", methods=['GET', 'POST'])
def chat():
    """
    {
        "messages": [
            {"role": "user", "content": "What is 300 times 1000"},
            {"role": "assistant", "content": "300 multiplied by 1000 is 300000."},
            {"role": "user", "content": "Who are you?"},
        ],
        "lang": "en",
    }
    """
    try:
        data = request.get_json()
        lang = data["lang"]
        if lang not in ["en", "ar"]:
            raise HTTPException(status_code=400, detail=f"Invalid language cod: '{lang}'")
        messages = data["messages"]
    except HTTPException:
        raise
    except Exception:
        raise abort(400)

    logging.info(f"request to send '{messages}' in language '{lang}'")
    try:
        start_time = time.time()
        res = chatbot.chat(messages)
        logging.debug(f"res: {res}")
        end_time = time.time()
        response = {"res": res, "time": end_time - start_time}
        return make_response(response, 200)
    except Exception as e:
        logging.error(e)
        raise abort(500)


if __name__ == "__main__":
    logging.info("Starting the AI chat service.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    if args.debug:
        logging.warning("It is now in the DEBUG mode.")
        log_model = logging.DEBUG
    else:
        log_model = logging.INFO
    logging.basicConfig(level=log_model)

    model_path = "/models/core42/jais-13b-chat/"
    if os.path.exists(model_path):
        logging.info(f"The path '{model_path}' will be loaded.")
    else:
        logging.error(f"Model path '{model_path}' is not exist!")

    chatbot = FastChatLLM(model_name=model_path, device='cuda', num_gpus='4', load_8bit=False, temperature=0.7, max_new_tokens=512, debug=False)

    app.run(debug=args.debug, host=args.host, port=args.port)
