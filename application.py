from flask import Flask
from spert.api.endpoint import keyphrases_bot

application = Flask(__name__)
application.config["JSON_SORT_KEYS"] = False

application.register_blueprint(keyphrases_bot)

if __name__ == "__main__":
    application.run(debug=False)
