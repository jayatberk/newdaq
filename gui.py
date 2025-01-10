import sys 
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QWidget, QTabWidget, QHBoxLayout, QGridLayout, QHeaderView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FileProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Processor GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Tabs setup
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab for charts
        self.chart_tab = QWidget()
        self.chart_layout = QGridLayout(self.chart_tab)
        self.tabs.addTab(self.chart_tab, "Charts")

        # Tab for table
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.tabs.addTab(self.table_tab, "Data Table")
        
        # Additional Tabs for future tables
        self.additional_table_tab_1 = QWidget()
        self.additional_table_tab_1_layout = QVBoxLayout(self.additional_table_tab_1)
        self.tabs.addTab(self.additional_table_tab_1, "Additional Table 1")

        self.additional_table_tab_2 = QWidget()
        self.additional_table_tab_2_layout = QVBoxLayout(self.additional_table_tab_2)
        self.tabs.addTab(self.additional_table_tab_2, "Additional Table 2")

        # Table setup for main table tab
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "Summary Statistics"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.table)

        # Table setup for additional table tabs (same structure as the main table)
        self.additional_table_1 = QTableWidget()
        self.additional_table_1.setColumnCount(4)
        self.additional_table_1.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "Summary Statistics"
        ])
        self.additional_table_1.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_1.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_1_layout.addWidget(self.additional_table_1)

        self.additional_table_2 = QTableWidget()
        self.additional_table_2.setColumnCount(4)
        self.additional_table_2.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "Summary Statistics"
        ])
        self.additional_table_2.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_2.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_2_layout.addWidget(self.additional_table_2)

        # Add File Button for Table Tab
        self.add_file_button_table = QPushButton("Add Files")
        self.add_file_button_table.clicked.connect(self.add_files)
        self.table_layout.addWidget(self.add_file_button_table)

        # Matplotlib Figures in a Quadrant Layout
        self.figures = [
            Figure(),
            Figure(),
            Figure(),
            Figure()
        ]
        self.canvases = [
            FigureCanvas(fig) for fig in self.figures
        ]

        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        self.chart_layout.addWidget(self.canvases[2], 1, 0)
        self.chart_layout.addWidget(self.canvases[3], 1, 1)

        # Add File Button for Charts Tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # Initialize plots
        self.initialize_plot(0)
        self.initialize_plot(1)
        self.initialize_plot(2)
        self.initialize_fft_plot()

        # Logic variables
        self.jkam_creation_time_array = []
        self.avg_time_gap = 0
        self.shots_num = 0
        self.start_time = None
        self.cumulative_accepted = 0
        self.cumulative_data = []

    def initialize_plot(self, index):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [], marker="o")
        ax.set_title(f"Cumulative Accepted Files {index + 1}")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.canvases[index].draw()

    def initialize_fft_plot(self):
        ax = self.figures[3].add_subplot(111)
        ax.plot([], [])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.canvases[3].draw()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "HDF5 Files (*.h5)")
        if not files:
            return

        for file in files:
            self.process_file(file)

        self.update_cumulative_plot()

    def process_file(self, file):
        file_ctime = os.path.getctime(file)
        self.jkam_creation_time_array.append(file_ctime)

        space_correct = True
        if self.shots_num == 0:
            self.start_time = file_ctime
        else:
            curr_time = file_ctime
            self.avg_time_gap = (curr_time - self.start_time) / self.shots_num
            time_temp = self.jkam_creation_time_array[self.shots_num]
            if (np.abs(time_temp - self.jkam_creation_time_array[self.shots_num - 1] - self.avg_time_gap) > 0.2 * self.avg_time_gap):
                space_correct = False

        self.cumulative_accepted += 1 if space_correct else 0
        self.shots_num += 1

        self.cumulative_data.append(self.cumulative_accepted if space_correct else 0)

        # Update Table
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(str(self.shots_num - 1)))
        self.table.setItem(row_position, 1, QTableWidgetItem(file))
        self.table.setItem(row_position, 2, QTableWidgetItem(str(space_correct)))
        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, <b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.table.setItem(row_position, 3, QTableWidgetItem(summary_text))

    def update_cumulative_plot(self):
        fig = self.figures[0]
        fig.clear()
        ax = fig.add_subplot(111)
        x_vals = list(range(len(self.cumulative_data)))
        ax.plot(x_vals, self.cumulative_data, marker="o", linestyle="-")
        ax.set_title("Cumulative Accepted Files 1")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.canvases[0].draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
