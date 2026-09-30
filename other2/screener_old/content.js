(() => {
    "use strict";

    const {
        AUTO_REFRESH_ENABLED,
        AUTO_REFRESH_INTERVAL_MS,
        TICKER_FIELD,
        PRICE_FIELD,
        PERCENT_CHANGE_FIELD
    } = SCREENER_CONFIG;


    const TABLE_BODY_SELECTOR =
        'tbody[data-testid="selectable-rows-table-body"]';

    const REFRESH_BUTTON_SELECTOR =
        'button[data-qa-id="screener-refresh-button"]';


    function parseNumber(text) {
        if (!text) {
            return NaN;
        }

        const normalized = text
            .replace(/[−–—]/g, "-")
            .replace(/,/g, "")
            .replace(/\u202F/g, " ")
            .trim();

        const match = normalized.match(/[-+]?\d*\.?\d+/);

        if (!match) {
            return NaN;
        }

        return Number(match[0]);
    }


    function getColumnIndexes() {
        const headerRow = document.querySelector("thead tr");

        if (!headerRow) {
            return null;
        }

        const headers = headerRow.querySelectorAll("th[data-field]");

        const indexes = {};

        headers.forEach((header, index) => {
            const field = header.getAttribute("data-field");

            if (field) {
                indexes[field] = index;
            }
        });

        if (
            indexes[TICKER_FIELD] === undefined ||
            indexes[PRICE_FIELD] === undefined ||
            indexes[PERCENT_CHANGE_FIELD] === undefined
        ) {
            return null;
        }

        return indexes;
    }


    function readScreener() {
        const rows = document.querySelectorAll(
            `${TABLE_BODY_SELECTOR} tr.listRow`
        );

        if (rows.length === 0) {
            return [];
        }

        const columnIndexes = getColumnIndexes();

        if (!columnIndexes) {
            return [];
        }

        const result = [];

        for (const row of rows) {
            const cells = row.querySelectorAll(":scope > td");

            const tickerCell =
                cells[columnIndexes[TICKER_FIELD]];

            const priceCell =
                cells[columnIndexes[PRICE_FIELD]];

            const percentChangeCell =
                cells[columnIndexes[PERCENT_CHANGE_FIELD]];

            if (
                !tickerCell ||
                !priceCell ||
                !percentChangeCell
            ) {
                continue;
            }

            const symbolElement =
                tickerCell.querySelector(
                    'a[class*="tickerNameBox"]'
                );

            if (!symbolElement) {
                continue;
            }

            const symbol =
                symbolElement.textContent.trim();

            const price =
                parseNumber(priceCell.textContent);

            const percentChange =
                parseNumber(percentChangeCell.textContent);

            if (
                !symbol ||
                !Number.isFinite(price) ||
                !Number.isFinite(percentChange)
            ) {
                continue;
            }

            result.push({
                symbol,
                price,
                percent_change: percentChange
            });
        }

        return result;
    }


    function clickRefreshButton() {
        if (!AUTO_REFRESH_ENABLED) {
            return;
        }

        const refreshButton = document.querySelector(
            'button[data-qa-id="screener-refresh-button"]'
        );

        if (!refreshButton) {
            console.log("[REFRESH] Button not found");
            return;
        }

        refreshButton.click();

        // console.log("[REFRESH] Clicked");
    }


    function sendUpdate() {
        const data = readScreener();

        if (data.length === 0) {
            return;
        }

        chrome.runtime.sendMessage({
            type: "SCREENER_UPDATE",
            data
        });
    }


    function update() {
        clickRefreshButton();
        sendUpdate();
    }


    update();


    if (AUTO_REFRESH_ENABLED) {
        setInterval(
            update,
            AUTO_REFRESH_INTERVAL_MS
        );
    }
})();