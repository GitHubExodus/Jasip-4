"use strict";

const PYTHON_URL = "http://127.0.0.1:8765/screener";

chrome.runtime.onMessage.addListener(
    (message, sender, sendResponse) => {

        if (message.type !== "SCREENER_UPDATE") {
            return;
        }

        fetch(PYTHON_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(message.data)
        })
        .then(response => {
            if (!response.ok) {
                console.error(
                    "[SCREENER → PYTHON] HTTP",
                    response.status
                );
                return;
            }

            console.log(
                "[SCREENER → PYTHON]",
                message.data.length,
                "stocks"
            );
        })
        .catch(error => {
            console.error(
                "[SCREENER → PYTHON]",
                error
            );
        });

        sendResponse({ ok: true });

        return true;
    }
);