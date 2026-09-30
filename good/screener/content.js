(() => {
    "use strict";

    const {
        AUTO_REFRESH_ENABLED,
        AUTO_REFRESH_INTERVAL_MS,
        TICKER_FIELD,
        DAILY_CHANGE_FIELD
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
            .replace(/\u00A0/g, " ")
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

        const headers = headerRow.querySelectorAll(
            "th[data-field]"
        );

        const columns = [];

        headers.forEach((header, index) => {
            const field = header.getAttribute("data-field");

            if (!field) {
                return;
            }

            columns.push({
                field,
                index
            });
        });

        if (columns.length === 0) {
            return null;
        }

        return columns;
    }


    function getCellValue(cell) {
        if (!cell) {
            return null;
        }

        const text = cell.textContent
            .replace(/\s+/g, " ")
            .trim();

        if (!text) {
            return null;
        }

        return text;
    }


    function getTickerData(tickerCell) {
        const symbolElement = tickerCell.querySelector(
            'a[class*="tickerNameBox"]'
        );

        const descriptionElement = tickerCell.querySelector(
            'a[class*="tickerDescription"]'
        );

        const symbol = symbolElement
            ? symbolElement.textContent.trim()
            : "";

        const name = descriptionElement
            ? descriptionElement.textContent.trim()
            : "";

        return {
            symbol,
            name
        };
    }


    function convertValue(field, text) {
        if (text === null) {
            return null;
        }

        if (field === TICKER_FIELD) {
            return text;
        }

        const number = parseNumber(text);

        if (Number.isFinite(number)) {
            return number;
        }

        return text;
    }


    function readScreener() {
        const rows = document.querySelectorAll(
            `${TABLE_BODY_SELECTOR} tr.listRow`
        );

        if (rows.length === 0) {
            return [];
        }

        const columns = getColumnIndexes();

        if (!columns) {
            return [];
        }

        const result = [];

        for (const row of rows) {
            const cells = row.querySelectorAll(
                ":scope > td"
            );

            if (cells.length === 0) {
                continue;
            }

            const tickerColumn = columns.find(
                column => column.field === TICKER_FIELD
            );

            if (!tickerColumn) {
                continue;
            }

            const tickerCell =
                cells[tickerColumn.index];

            if (!tickerCell) {
                continue;
            }

            const tickerData =
                getTickerData(tickerCell);

            if (!tickerData.symbol) {
                continue;
            }

            const stock = {
                symbol: tickerData.symbol,
                name: tickerData.name
            };

            for (const column of columns) {
                const cell = cells[column.index];

                const text = getCellValue(cell);

                stock[column.field] =
                    convertValue(
                        column.field,
                        text
                    );
            }

            result.push(stock);
        }

        result.sort((a, b) => {
            const aChange =
                Number(a[DAILY_CHANGE_FIELD]);

            const bChange =
                Number(b[DAILY_CHANGE_FIELD]);

            if (!Number.isFinite(aChange)) {
                return 1;
            }

            if (!Number.isFinite(bChange)) {
                return -1;
            }

            return bChange - aChange;
        });

        return result;
    }


    function clickRefreshButton() {
        if (!AUTO_REFRESH_ENABLED) {
            return;
        }

        const refreshButton = document.querySelector(
            REFRESH_BUTTON_SELECTOR
        );

        if (!refreshButton) {
            console.log(
                "[REFRESH] Button not found"
            );

            return;
        }

        refreshButton.click();
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