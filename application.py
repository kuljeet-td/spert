from flask import Flask, jsonify
from spert.api.endpoint import keyphrases_bot

application = Flask(__name__)
application.config["JSON_SORT_KEYS"] = False

application.register_blueprint(keyphrases_bot)


@application.route("/")
def index():
    return jsonify({"message": "You see this message. The application is running on 8082."})


if __name__ == "__main__":
    application.run(debug=False)
