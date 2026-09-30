from __future__ import annotations

from flask import Flask, jsonify, request

from trading_system import TradingSystem


app = Flask(__name__)


API_KEY = "PKA4A6THLEKI6QD2MQPOAO25J3"
SECRET_KEY = "4nj9w53vMrNKJGZsHqN7Siqy34z2Gis9TffWi2beszNU"


trading_system = TradingSystem(
    alpaca_api_key=API_KEY,
    alpaca_secret_key=SECRET_KEY,
    paper=True,
)

trading_system.start()


@app.post("/screener")
def screener_update():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({
            "ok": False,
            "error": "Expected a JSON list.",
        }), 400

    try:
        trading_system.process_screener_update(
            data
        )

        return jsonify({
            "ok": True,
            "stocks": len(data),
        })

    except Exception as error:
        print(
            f"[SCREENER ERROR] {error}"
        )

        return jsonify({
            "ok": False,
            "error": str(error),
        }), 500


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8765,
        debug=False,
        threaded=True,
    )