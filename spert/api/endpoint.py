import time
from flask import request, jsonify, Blueprint
from spert.main import __predict

keyphrases_bot = Blueprint("app", __name__, url_prefix="/keyphrase_extraction")


def text_formulation(string_):
    return [string_.split(" ")]


@keyphrases_bot.route('/spert', methods=['POST'])
def recommendation_fetch():
    """
    candidate/companies recommendations in json format
    """
    try:
        payload_data_ = request.get_json()
        text_ = payload_data_['text']
        data = text_formulation(text_)
        st = time.time()
        resp, entity = __predict(data)
        time_ = time.time() - st
        return jsonify(
            {'data': resp, 'predictions': entity, 'time taken by api': time_, 'status_code': 200})
    except Exception as e:
        return jsonify({'error': 'Some error has been encountered, Please try after sometime'})
