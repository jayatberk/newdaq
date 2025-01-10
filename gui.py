import sys 
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QWidget, QTabWidget, QHBoxLayout, QGridLayout, QHeaderView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class H5FileHandler:
    def __init__(self, gui):
        self.gui = gui
        self.shots_num = 0
        self.cumulative_data = []
        self.start_time = None
        self.avg_time_gap = 0
        self.jkam_creation_time_array = []
        self.all_datapoints = []  # Store all data points for FFT

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

        if space_correct:
            self.cumulative_data.append(self.cumulative_data[-1] + 1 if self.cumulative_data else 1)
        else:
            self.cumulative_data.append(0)

        self.shots_num += 1

        # Simulate appending a datapoint (for example purposes)
        self.all_datapoints.append(file_ctime)  # Replace with actual datapoint extraction logic

        # Update Table
        row_position = self.gui.table.rowCount()
        self.gui.table.insertRow(row_position)
        self.gui.table.setItem(row_position, 0, QTableWidgetItem(str(self.shots_num - 1)))
        self.gui.table.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.table.setItem(row_position, 2, QTableWidgetItem(str(space_correct)))
        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, <b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.table.setItem(row_position, 3, QTableWidgetItem(summary_text))

        self.update_cumulative_plot()
        self.update_fft_plot()

    def update_cumulative_plot(self):
        fig = self.gui.figures[0]
        fig.clear()
        ax = fig.add_subplot(111)
        x_vals = list(range(len(self.cumulative_data)))
        ax.plot(x_vals, self.cumulative_data, marker="o", linestyle="-")
        ax.set_title("Cumulative Accepted Files 1")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[0].draw()

    def update_fft_plot(self):
        if len(self.all_datapoints) < 2:
            return

        fig = self.gui.figures[2]
        fig.clear()
        ax = fig.add_subplot(111)

        # Perform FFT
        fft_result = np.fft.fft(self.all_datapoints)
        freqs = np.fft.fftfreq(len(self.all_datapoints))

        # Plot FFT
        ax.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.gui.canvases[2].draw()

class BinFileHandler:
    def __init__(self, gui):
        self.gui = gui
        self.shots_num = 0
        self.cumulative_data = []
        self.start_time = None
        self.avg_time_gap = 0
        self.fpga_creation_time_array = []

    def process_file(self, file):
        # Simulating processing logic for .bin files
        file_ctime = os.path.getctime(file)
        self.fpga_creation_time_array.append(file_ctime)

        data_valid = True
        if self.shots_num == 0:
            self.start_time = file_ctime
        else:
            curr_time = file_ctime
            self.avg_time_gap = (curr_time - self.start_time) / self.shots_num

        if data_valid:
            self.cumulative_data.append(self.cumulative_data[-1] + 1 if self.cumulative_data else 1)
        else:
            self.cumulative_data.append(0)

        self.shots_num += 1

        # Update Table 2
        row_position = self.gui.additional_table_1.rowCount()
        self.gui.additional_table_1.insertRow(row_position)
        self.gui.additional_table_1.setItem(row_position, 0, QTableWidgetItem(str(self.shots_num - 1)))
        self.gui.additional_table_1.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.additional_table_1.setItem(row_position, 2, QTableWidgetItem(str(data_valid)))
        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, <b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.additional_table_1.setItem(row_position, 3, QTableWidgetItem(summary_text))

        self.update_chart_2()

    def update_chart_2(self):
        fig = self.gui.figures[1]
        fig.clear()
        ax = fig.add_subplot(111)
        x_vals = list(range(len(self.cumulative_data)))
        ax.plot(x_vals, self.cumulative_data, marker="o", linestyle="-")
        ax.set_title("Cumulative Accepted Files 2")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[1].draw()

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
        
        # Additional Tab for another table
        self.additional_table_tab_1 = QWidget()
        self.additional_table_tab_1_layout = QVBoxLayout(self.additional_table_tab_1)
        self.tabs.addTab(self.additional_table_tab_1, "Additional Table 1")

        # Table setup for main table tab
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "Summary Statistics"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.table)

        # Table setup for additional table tab (same structure as the main table)
        self.additional_table_1 = QTableWidget()
        self.additional_table_1.setColumnCount(4)
        self.additional_table_1.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "Summary Statistics"
        ])
        self.additional_table_1.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_1.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_1_layout.addWidget(self.additional_table_1)

        # Add File Button for Table Tab
        self.add_file_button_table = QPushButton("Add Files")
        self.add_file_button_table.clicked.connect(self.add_files)
        self.table_layout.addWidget(self.add_file_button_table)

        # Matplotlib Figures in a Quadrant Layout
        self.figures = [
            Figure(),
            Figure(),
            Figure()
        ]
        self.canvases = [
            FigureCanvas(fig) for fig in self.figures
        ]

        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        self.chart_layout.addWidget(self.canvases[2], 1, 0)  # FFT moved to bottom left

        # Add File Button for Charts Tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # Initialize plots
        self.initialize_plot(0)
        self.initialize_plot(1)
        self.initialize_fft_plot()

        # File Handlers
        self.h5_handler = H5FileHandler(self)
        self.bin_handler = BinFileHandler(self)

    def initialize_plot(self, index):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [], marker="o")
        ax.set_title(f"Cumulative Accepted Files {index + 1}")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.canvases[index].draw()

    def initialize_fft_plot(self):
        ax = self.figures[2].add_subplot(111)
        ax.plot([], [])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.canvases[2].draw()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*.*)")
        if not files:
            return

        for file in files:
            file_extension = os.path.splitext(file)[-1].lower()
            if file_extension == ".h5":
                self.h5_handler.process_file(file)
            elif file_extension == ".bin":
                self.bin_handler.process_file(file)
            else:
                print(f"Unsupported file type: {file_extension}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
