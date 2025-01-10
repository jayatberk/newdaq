import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QWidget, QTabWidget, QGridLayout, QHeaderView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class JkamH5FileHandler:
    def __init__(self, gui):
        self.gui = gui
        self.jkam_files = []  # We'll store the files for reference if needed

        # Data arrays
        self.jkam_creation_time_array = []  # Creation times for each shot
        self.shots_dict = {}                # {shot_index: space_correct_boolean}
        self.time_temp_dict = {}            # {shot_index: time_temp_value}

        # Tracking
        self.shots_num = 0
        self.start_time = None
        self.avg_time_gap = 0

        # For the first chart (Cumulative Data)
        self.cumulative_data = []

        # For the FFT chart
        self.all_datapoints = []  # Could store data from each shot for FFT

    def process_file(self, file):
        """
        Process a single JKAM .h5 file.
        We'll treat the creation time as time_temp.
        """
        file_ctime = os.path.getctime(file)
        self.jkam_files.append(file)
        self.jkam_creation_time_array.append(file_ctime)

        space_correct = True
        time_temp = file_ctime

        if self.shots_num == 0:
            self.start_time = file_ctime
        else:
            curr_time = file_ctime
            # average gap is from start_time across #shots so far
            self.avg_time_gap = (curr_time - self.start_time) / self.shots_num
            prev_time_temp = self.jkam_creation_time_array[self.shots_num - 1]

            # Check if this shot’s time is near the expected spacing
            if (abs(time_temp - prev_time_temp - self.avg_time_gap) > 0.2 * self.avg_time_gap):
                space_correct = False

        # Store space_correct & time_temp for this shot index
        self.shots_dict[self.shots_num] = space_correct
        self.time_temp_dict[self.shots_num] = time_temp

        # Append to cumulative chart
        if space_correct:
            self.cumulative_data.append(self.cumulative_data[-1] + 1 if self.cumulative_data else 1)
        else:
            self.cumulative_data.append(0)

        self.shots_num += 1

        # For FFT example, store the creation time in all_datapoints
        self.all_datapoints.append(file_ctime)

        # Update the GUI table
        row_position = self.gui.table.rowCount()
        self.gui.table.insertRow(row_position)
        self.gui.table.setItem(row_position, 0, QTableWidgetItem(str(self.shots_num - 1)))
        self.gui.table.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.table.setItem(row_position, 2, QTableWidgetItem(str(space_correct)))
        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, "
            f"<b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.table.setItem(row_position, 3, QTableWidgetItem(summary_text))

        # Update chart/FFT
        self.update_cumulative_plot()
        self.update_fft_plot()

    def update_cumulative_plot(self):
        fig = self.gui.figures[0]
        fig.clear()
        ax = fig.add_subplot(111)
        x_vals = list(range(len(self.cumulative_data)))
        ax.plot(x_vals, self.cumulative_data, marker="o", linestyle="-")
        ax.set_title("Cumulative Accepted Files 1 (JKAM)")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[0].draw()

    def update_fft_plot(self):
        """
        Simple example to show an FFT. 
        We'll only run if we have at least 2 data points.
        """
        if len(self.all_datapoints) < 2:
            return

        fig = self.gui.figures[2]
        fig.clear()
        ax = fig.add_subplot(111)

        # Perform FFT of creation times (just a sample demonstration!)
        fft_result = np.fft.fft(self.all_datapoints)
        freqs = np.fft.fftfreq(len(self.all_datapoints))

        ax.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.gui.canvases[2].draw()


class GageScopeH5FileHandler:
    def __init__(self, gui):
        self.gui = gui
        # If you need to handle GageScope, place logic here

    def process_file(self, file):
        # Stub method
        pass


class BinFileHandler:
    def __init__(self, gui):
        self.gui = gui

        # For .bin files
        self.bin_files = []
        self.fpga_creation_time_array = []

        # Acceptance logic arrays
        self.mask_valid_data = []     # True/False for each shot
        self.jkam_fpga_matchlist = [] # Index of matching JKAM shot or -1
        self.cumulative_data = []     # For the 2nd chart

        # Tracking
        self.start_time = None
        self.avg_time_gap = 0

    def process_file(self, file):
        """
        Each time we get a new bin file, let's store it, then we will
        re-run acceptance logic for the entire bin-file list from scratch.
        """
        file_ctime = os.path.getctime(file)
        self.bin_files.append(file)
        self.fpga_creation_time_array.append(file_ctime)

        # If it's our first bin shot, define the start_time
        if len(self.fpga_creation_time_array) == 1:
            self.start_time = file_ctime

        # Re-compute the entire acceptance logic for ALL shots
        self.rerun_acceptance()

        # Finally, update the table with the info for this single newly added file
        new_shot_index = len(self.fpga_creation_time_array) - 1
        data_valid = self.mask_valid_data[new_shot_index] if 0 <= new_shot_index < len(self.mask_valid_data) else False

        row_position = self.gui.additional_table_1.rowCount()
        self.gui.additional_table_1.insertRow(row_position)
        self.gui.additional_table_1.setItem(row_position, 0, QTableWidgetItem(str(new_shot_index)))
        self.gui.additional_table_1.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.additional_table_1.setItem(row_position, 2, QTableWidgetItem(str(data_valid)))

        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, "
            f"<b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.additional_table_1.setItem(row_position, 3, QTableWidgetItem(summary_text))

        # Update the 2nd chart (Cumulative acceptance)
        self.update_chart_2()

    def rerun_acceptance(self):
        """
        Clear out all acceptance logic results and re-run from scratch.
        The shot index for each bin file is simply 0..n-1,
        and we'll attempt to match it with JKAM data of the same index.
        """
        num_shots = len(self.fpga_creation_time_array)

        # If we have 0 or 1 shots, no meaningful avg_time_gap
        if num_shots <= 1:
            self.avg_time_gap = 0
        else:
            # Use the total span from first shot to last shot / (num_shots-1)
            total_span = self.fpga_creation_time_array[-1] - self.fpga_creation_time_array[0]
            self.avg_time_gap = total_span / (num_shots - 1)

        # Initialize arrays
        self.mask_valid_data = np.zeros(num_shots, dtype=bool)
        self.jkam_fpga_matchlist = np.zeros(num_shots, dtype=int) - 1

        # We'll rebuild the cumulative_data array
        self.cumulative_data = []

        # Retrieve JKAM dictionaries
        jkam_space_dict = self.gui.jkam_h5_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_handler.time_temp_dict

        # Convert to numpy for vectorized checks
        fpga_ctimes = np.array(self.fpga_creation_time_array)
        fpga_index_list = np.arange(num_shots)

        # For each shot index in bin files, see if we have JKAM data
        for shot_num in range(num_shots):
            # Only if JKAM data for this shot_num exists
            if shot_num in jkam_time_temp_dict and shot_num in jkam_space_dict:
                jkam_time = jkam_time_temp_dict[shot_num]
                space_correct = jkam_space_dict[shot_num]

                # Condition: is there an FPGA ctime within 0.2 * avg_time_gap of jkam_time?
                # Also must check that jkam_space_dict[shot_num] == True
                time_diffs = np.abs(fpga_ctimes - jkam_time)
                min_diff = np.min(time_diffs)

                if (min_diff <= 0.2 * self.avg_time_gap) and space_correct:
                    self.mask_valid_data[shot_num] = True
                    closest_idx = np.argmin(time_diffs)
                    self.jkam_fpga_matchlist[shot_num] = fpga_index_list[closest_idx]
                else:
                    # If we fail the conditions, mark it invalid
                    self.mask_valid_data[shot_num] = False
                    self.jkam_fpga_matchlist[shot_num] = -1
                    print(f"error at shot {shot_num}")
            else:
                # No JKAM data for this shot yet => do nothing
                self.mask_valid_data[shot_num] = False
                self.jkam_fpga_matchlist[shot_num] = -1

            # Build the cumulative_data array based on acceptance
            # If we accepted this shot, increment; else 0
            if self.mask_valid_data[shot_num]:
                new_val = (self.cumulative_data[-1] + 1) if self.cumulative_data else 1
            else:
                new_val = 0
            self.cumulative_data.append(new_val)

    def update_chart_2(self):
        """
        Plot the cumulative acceptance of bin files in the second chart.
        We'll plot the jkam_fpga_matchlist acceptance results.
        """
        fig = self.gui.figures[1]
        fig.clear()
        ax = fig.add_subplot(111)
        x_vals = list(range(len(self.cumulative_data)))
        ax.plot(x_vals, self.cumulative_data, marker="o", linestyle="-")
        ax.set_title("Cumulative Accepted Files 2 (Bin)")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[1].draw()


class FileProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Processor GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.layout = QVBoxLayout(self.central_widget)

        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Chart tab
        self.chart_tab = QWidget()
        self.chart_layout = QGridLayout(self.chart_tab)
        self.tabs.addTab(self.chart_tab, "Charts")

        # Table tab (JKAM table)
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.tabs.addTab(self.table_tab, "Data Table")

        # Additional table tab (Bin table)
        self.additional_table_tab_1 = QWidget()
        self.additional_table_tab_1_layout = QVBoxLayout(self.additional_table_tab_1)
        self.tabs.addTab(self.additional_table_tab_1, "Additional Table 1")

        # Set up JKAM table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Shot Number", "File Name", "Accepted", "Summary Statistics"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.table)

        # Set up Bin table
        self.additional_table_1 = QTableWidget()
        self.additional_table_1.setColumnCount(4)
        self.additional_table_1.setHorizontalHeaderLabels(["Shot Number", "File Name", "Accepted", "Summary Statistics"])
        self.additional_table_1.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_1.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_1_layout.addWidget(self.additional_table_1)

        # File button in table tab
        self.add_file_button_table = QPushButton("Add Files")
        self.add_file_button_table.clicked.connect(self.add_files)
        self.table_layout.addWidget(self.add_file_button_table)

        # Setup the 3 figures
        self.figures = [Figure(), Figure(), Figure()]
        self.canvases = [FigureCanvas(fig) for fig in self.figures]

        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        self.chart_layout.addWidget(self.canvases[2], 1, 0)

        # File button in charts tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # Initialize plots
        self.initialize_plot(0, "Cumulative Accepted Files 1 (JKAM)")
        self.initialize_plot(1, "Cumulative Accepted Files 2 (Bin)")
        self.initialize_fft_plot()

        # Handlers
        self.jkam_h5_handler = JkamH5FileHandler(self)
        self.gagescope_h5_handler = GageScopeH5FileHandler(self)
        self.bin_handler = BinFileHandler(self)

    def initialize_plot(self, index, title_str):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [], marker="o")
        ax.set_title(title_str)
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
        """
        Common file-adding function used by both the table and charts tab buttons.
        """
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*.*)")
        if not files:
            return

        for file in files:
            file_extension = os.path.splitext(file)[-1].lower()

            if file_extension == ".h5":
                if "jkam" in file.lower():
                    self.jkam_h5_handler.process_file(file)
                elif "gage" in file.lower():
                    self.gagescope_h5_handler.process_file(file)
                else:
                    print("Unsupported .h5 file (not recognized as JKAM or GageScope).")
            elif file_extension == ".bin":
                self.bin_handler.process_file(file)
            else:
                print(f"Unsupported file type: {file_extension}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
