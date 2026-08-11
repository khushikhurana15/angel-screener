import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QColor
import data_engine as de

COLUMNS = [
    "Symbol", "LTP", "SMMA(20)", "SMMA(120)", "Signal",
    "ETQ(5m)", "ETQ(20m)", "ETQ(60m)", "Avg(20m)", "Avg(60m)",
    "Bid Price", "Bid Qty", "Ask Price", "Ask Qty",
    "ML Pred", "Confidence", "Explanation"
]


class DataWorker(QThread):
    row_ready = Signal(dict)
    row_disqualified = Signal(str)
    status_update = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            print("LOGIN: starting...")
            de.login()
            print("LOGIN: done")
            de.load_cache_from_disk()
            de.load_model()
            print("MODEL: loaded")
            de.load_groq()
            print("GROQ: loaded")
            stocks = de.load_qualified_stocks()
            tokens = [str(s["token"]) for s in stocks]
            print(f"STOCKS: loaded {len(stocks)} stocks")
        except Exception as e:
            self.status_update.emit(f"Startup failed: {e}")
            print(f"STARTUP ERROR: {e}")
            return

        while self._running:
            try:
                self.status_update.emit("Fetching market depth batch...")
                print("DEPTH: fetching...")
                depth_map = de.get_market_depth_batch(tokens)
                print(f"DEPTH: got {len(depth_map)} entries")

                for stock in stocks:
                    if not self._running:
                        break

                    symbol, token = stock["symbol"], str(stock["token"])
                    depth_info = depth_map.get(token, {})

                    if not de.still_qualifies(depth_info):
                        self.row_disqualified.emit(symbol)
                        continue

                    self.status_update.emit(f"Updating {symbol}...")
                    print(f"PROCESSING: {symbol}")

                    is_cold_start = token not in de._candle_cache
                    row = de.build_stock_snapshot(symbol, token, depth_info)
                    if row:
                        self.row_ready.emit(row)
                    print(f"DONE: {symbol}")

                    time.sleep(2.5 if is_cold_start else 1.5)

                de.save_cache_to_disk()
            except Exception as e:
                print(f"LOOP ERROR: {e}")
                self.status_update.emit(f"Error: {e}")
                time.sleep(5)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI/ML Stock Screener Dashboard")
        self.setGeometry(50, 50, 1800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.status_label = QLabel("Status: Starting...")
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        layout.addWidget(self.table)

        self.stock_data = {}

        self.worker = DataWorker()
        self.worker.row_ready.connect(self.update_row)
        self.worker.row_disqualified.connect(self.remove_row)
        self.worker.status_update.connect(self.update_status)
        self.worker.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.repaint_table)
        self.timer.start(2000)

    def update_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def update_row(self, row_data):
        symbol = row_data["Symbol"]
        self.stock_data[symbol] = row_data

    def remove_row(self, symbol):
        if symbol in self.stock_data:
            del self.stock_data[symbol]

    def repaint_table(self):
        self.table.setRowCount(len(self.stock_data))
        for row_idx, (symbol, row_data) in enumerate(self.stock_data.items()):
            for col_idx, col_name in enumerate(COLUMNS):
                value = str(row_data.get(col_name, "-"))
                item = QTableWidgetItem(value)

                if col_name == "Signal":
                    if value == "BUY":
                        item.setBackground(QColor("#2e7d32"))
                    elif value == "SELL":
                        item.setBackground(QColor("#c62828"))

                if col_name == "ML Pred":
                    if value == "Profitable":
                        item.setForeground(QColor("#2e7d32"))
                    elif value == "Avoid":
                        item.setForeground(QColor("#c62828"))

                self.table.setItem(row_idx, col_idx, item)

        self.table.resizeColumnsToContents()

    def closeEvent(self, event):
        self.status_label.setText("Status: Shutting down...")
        self.worker.stop()
        self.worker.wait(3000)
        event.accept()


app = QApplication(sys.argv)
window = DashboardWindow()
window.show()
sys.exit(app.exec())