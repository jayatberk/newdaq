import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QWidget, QTabWidget, QGridLayout, QHeaderView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

###############################################################################
#                          JKAM Handler                                       #
###############################################################################
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
        try:
            file_ctime = os.path.getctime(file)
        except Exception as e:
            print(f"Error accessing file time for {file}: {e}")
            return

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

        # Update the GUI table (the main JKAM table)
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

        # Perform FFT of creation times (just a demonstration!)
        fft_result = np.fft.fft(self.all_datapoints)
        freqs = np.fft.fftfreq(len(self.all_datapoints))

        ax.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.gui.canvases[2].draw()

###############################################################################
#                          FPGA / Bin Handler                                 #
###############################################################################
class BinFileHandler:
    """
    Handles FPGA .bin files with acceptance logic against JKAM data.
    """
    def __init__(self, gui):
        self.gui = gui

        # For .bin (FPGA) files
        self.bin_files = []
        self.fpga_creation_time_array = []

        # Acceptance logic arrays
        self.mask_valid_data = []
        self.jkam_fpga_matchlist = []
        self.cumulative_data = []

        # A color array to mark points red (no JKAM) or green (JKAM found)
        self.color_array = []

        # Tracking
        self.start_time = None
        self.avg_time_gap = 0

    def process_file(self, file):
        """
        Each time we get a new bin file, let's store it, then we will
        re-run acceptance logic for the entire bin-file list from scratch.
        """
        try:
            file_ctime = os.path.getctime(file)
        except Exception as e:
            print(f"Error accessing file time for {file}: {e}")
            return

        self.bin_files.append(file)
        self.fpga_creation_time_array.append(file_ctime)

        # If it's our first bin shot, define the start_time
        if len(self.fpga_creation_time_array) == 1:
            self.start_time = file_ctime

        # Re-compute the entire acceptance logic for ALL shots
        self.rerun_acceptance()

        # Finally, update the table with the info for this single newly added file
        new_shot_index = len(self.fpga_creation_time_array) - 1
        data_valid = False
        jkam_space_correct_str = "None"

        if 0 <= new_shot_index < len(self.mask_valid_data):
            data_valid = self.mask_valid_data[new_shot_index]
        # If JKAM existed for that shot, show whether it was space-correct
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        if new_shot_index in jkam_space_dict:
            jkam_space_correct_str = str(jkam_space_dict[new_shot_index])

        # Add row to Additional Table 1
        row_position = self.gui.additional_table_1.rowCount()
        self.gui.additional_table_1.insertRow(row_position)
        self.gui.additional_table_1.setItem(row_position, 0, QTableWidgetItem(str(new_shot_index)))
        self.gui.additional_table_1.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.additional_table_1.setItem(row_position, 2, QTableWidgetItem(str(data_valid)))
        self.gui.additional_table_1.setItem(row_position, 3, QTableWidgetItem(jkam_space_correct_str))

        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, "
            f"<b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.additional_table_1.setItem(row_position, 4, QTableWidgetItem(summary_text))

        # Update the 2nd chart (Cumulative acceptance)
        self.update_chart_2()

    def rerun_acceptance(self):
        """
        Clear out all acceptance logic results and re-run from scratch.
        The shot index for each bin file is simply 0..n-1,
        and we'll attempt to match it with JKAM data of the same index.
        """
        num_shots = len(self.fpga_creation_time_array)

        if num_shots <= 1:
            self.avg_time_gap = 0
        else:
            total_span = self.fpga_creation_time_array[-1] - self.fpga_creation_time_array[0]
            self.avg_time_gap = total_span / (num_shots - 1)

        # Initialize arrays
        self.mask_valid_data = np.zeros(num_shots, dtype=bool)
        self.jkam_fpga_matchlist = np.zeros(num_shots, dtype=int) - 1
        self.cumulative_data = []
        self.color_array = ["r"] * num_shots  # Default = red if no JKAM

        # Retrieve JKAM dictionaries
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict

        # Convert to numpy
        fpga_ctimes = np.array(self.fpga_creation_time_array)
        fpga_index_list = np.arange(num_shots)

        for shot_num in range(num_shots):
            if shot_num in jkam_time_temp_dict and shot_num in jkam_space_dict:
                jkam_time = jkam_time_temp_dict[shot_num]
                space_correct = jkam_space_dict[shot_num]

                if self.avg_time_gap == 0:
                    # If there's only one shot, just accept if JKAM says space_correct
                    if space_correct:
                        self.mask_valid_data[shot_num] = True
                        self.color_array[shot_num] = "g"
                        self.jkam_fpga_matchlist[shot_num] = shot_num
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.color_array[shot_num] = "r"
                        self.jkam_fpga_matchlist[shot_num] = -1
                else:
                    time_diffs = np.abs(fpga_ctimes - jkam_time)
                    min_diff = np.min(time_diffs)

                    # Accept if min_diff <= 0.2 * avg_time_gap AND JKAM is space_correct
                    if (min_diff <= 0.2 * self.avg_time_gap) and space_correct:
                        self.mask_valid_data[shot_num] = True
                        closest_idx = np.argmin(time_diffs)
                        self.jkam_fpga_matchlist[shot_num] = fpga_index_list[closest_idx]
                        self.color_array[shot_num] = "g"  # green if matched
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.jkam_fpga_matchlist[shot_num] = -1
                        self.color_array[shot_num] = "r"
                        print(f"FPGA error at shot {shot_num}")
            else:
                # If no JKAM data, remain red & mask = False
                self.mask_valid_data[shot_num] = False
                self.jkam_fpga_matchlist[shot_num] = -1

            # Build the cumulative_data array based on acceptance
            if self.mask_valid_data[shot_num]:
                new_val = (self.cumulative_data[-1] + 1) if self.cumulative_data else 1
            else:
                new_val = 0
            self.cumulative_data.append(new_val)

    def update_chart_2(self):
        """
        Plot the cumulative acceptance of bin (FPGA) files in the second chart.
        Points are green if JKAM exists & accepted, red if JKAM doesn't exist or not accepted.
        """
        fig = self.gui.figures[1]
        fig.clear()
        ax = fig.add_subplot(111)

        x_vals = np.arange(len(self.cumulative_data))
        # Plot each point with the color decided in self.color_array
        for i in range(len(self.cumulative_data)):
            ax.plot(x_vals[i], self.cumulative_data[i], marker="o", color=self.color_array[i])

        # Optional: connect points with a line
        ax.plot(x_vals, self.cumulative_data, linestyle="-", alpha=0.3)

        ax.set_title("Cumulative Accepted Files 2 (Bin/FPGA)")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[1].draw()

###############################################################################
#                      GageScope .h5 Handler                                  #
###############################################################################
class GageScopeH5FileHandler:
    """
    Handles GageScope .h5 files with the same acceptance logic as Bin/FPGA.
    """
    def __init__(self, gui):
        self.gui = gui

        self.gage_files = []
        self.gage_creation_time_array = []

        # Acceptance logic arrays
        self.mask_valid_data = []
        self.jkam_gage_matchlist = []
        self.cumulative_data = []
        self.color_array = []

        # Tracking
        self.start_time = None
        self.avg_time_gap = 0

    def process_file(self, file):
        try:
            file_ctime = os.path.getctime(file)
        except Exception as e:
            print(f"Error accessing file time for {file}: {e}")
            return

        self.gage_files.append(file)
        self.gage_creation_time_array.append(file_ctime)

        if len(self.gage_creation_time_array) == 1:
            self.start_time = file_ctime

        # Rerun acceptance for all existing Gage shots
        self.rerun_acceptance_gage()

        # Insert new row into Additional Table 2
        new_shot_index = len(self.gage_creation_time_array) - 1
        data_valid = False
        jkam_space_correct_str = "None"

        if 0 <= new_shot_index < len(self.mask_valid_data):
            data_valid = self.mask_valid_data[new_shot_index]

        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        if new_shot_index in jkam_space_dict:
            jkam_space_correct_str = str(jkam_space_dict[new_shot_index])

        row_position = self.gui.additional_table_2.rowCount()
        self.gui.additional_table_2.insertRow(row_position)
        self.gui.additional_table_2.setItem(row_position, 0, QTableWidgetItem(str(new_shot_index)))
        self.gui.additional_table_2.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.additional_table_2.setItem(row_position, 2, QTableWidgetItem(str(data_valid)))
        self.gui.additional_table_2.setItem(row_position, 3, QTableWidgetItem(jkam_space_correct_str))

        summary_text = (
            f"<b>Start Time:</b> {self.start_time}, "
            f"<b>Current Time:</b> {file_ctime}, "
            f"<b>Avg Time Gap:</b> {self.avg_time_gap}"
        )
        self.gui.additional_table_2.setItem(row_position, 4, QTableWidgetItem(summary_text))

        self.update_chart_3()

    def rerun_acceptance_gage(self):
        num_shots = len(self.gage_creation_time_array)
        if num_shots <= 1:
            self.avg_time_gap = 0
        else:
            total_span = self.gage_creation_time_array[-1] - self.gage_creation_time_array[0]
            self.avg_time_gap = total_span / (num_shots - 1)

        # Initialize arrays
        self.mask_valid_data = np.zeros(num_shots, dtype=bool)
        self.jkam_gage_matchlist = np.zeros(num_shots, dtype=int) - 1
        self.cumulative_data = []
        self.color_array = ["r"] * num_shots  # default red if no JKAM

        # Retrieve JKAM dictionaries
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict

        gage_ctimes = np.array(self.gage_creation_time_array)
        gage_index_list = np.arange(num_shots)

        for shot_num in range(num_shots):
            if shot_num in jkam_time_temp_dict and shot_num in jkam_space_dict:
                jkam_time = jkam_time_temp_dict[shot_num]
                space_correct = jkam_space_dict[shot_num]

                if self.avg_time_gap == 0:
                    # If there's only one Gage shot, accept if JKAM is space_correct
                    if space_correct:
                        self.mask_valid_data[shot_num] = True
                        self.color_array[shot_num] = "g"
                        self.jkam_gage_matchlist[shot_num] = shot_num
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.color_array[shot_num] = "r"
                        self.jkam_gage_matchlist[shot_num] = -1
                else:
                    time_diffs = np.abs(gage_ctimes - jkam_time)
                    min_diff = np.min(time_diffs)

                    if (min_diff <= 0.2 * self.avg_time_gap) and space_correct:
                        self.mask_valid_data[shot_num] = True
                        closest_idx = np.argmin(time_diffs)
                        self.jkam_gage_matchlist[shot_num] = gage_index_list[closest_idx]
                        self.color_array[shot_num] = "g"
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.jkam_gage_matchlist[shot_num] = -1
                        self.color_array[shot_num] = "r"
                        print(f"Gage error at shot {shot_num}")
            else:
                # No JKAM data => remain red
                self.mask_valid_data[shot_num] = False
                self.jkam_gage_matchlist[shot_num] = -1

            if self.mask_valid_data[shot_num]:
                new_val = (self.cumulative_data[-1] + 1) if self.cumulative_data else 1
            else:
                new_val = 0
            self.cumulative_data.append(new_val)

    def update_chart_3(self):
        fig = self.gui.figures[3]
        fig.clear()
        ax = fig.add_subplot(111)

        x_vals = np.arange(len(self.cumulative_data))
        for i in range(len(self.cumulative_data)):
            ax.plot(x_vals[i], self.cumulative_data[i], marker="o", color=self.color_array[i])
        ax.plot(x_vals, self.cumulative_data, linestyle="-", alpha=0.3)

        ax.set_title("Cumulative Accepted Files 3 (GageScope)")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[3].draw()

###############################################################################
#                           Main GUI                                          #
###############################################################################
class FileProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Processor GUI")
        # Widen to accommodate 4 charts + extra tables
        self.setGeometry(100, 100, 1400, 800)

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
        self.tabs.addTab(self.table_tab, "JKAM Data Table")

        # Additional table tab 1 (FPGA/Bin)
        self.additional_table_tab_1 = QWidget()
        self.additional_table_tab_1_layout = QVBoxLayout(self.additional_table_tab_1)
        self.tabs.addTab(self.additional_table_tab_1, "Additional Table 1 (FPGA)")

        # Additional table tab 2 (GageScope)
        self.additional_table_tab_2 = QWidget()
        self.additional_table_tab_2_layout = QVBoxLayout(self.additional_table_tab_2)
        self.tabs.addTab(self.additional_table_tab_2, "Additional Table 2 (GageScope)")

        # Set up JKAM table (4 columns)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Shot Number", "File Name", "Accepted", "Summary Statistics"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.table)

        # Set up Bin table (FPGA) with 5 columns
        self.additional_table_1 = QTableWidget()
        self.additional_table_1.setColumnCount(5)
        self.additional_table_1.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "JKAM Space Correct", "Summary Statistics"
        ])
        self.additional_table_1.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_1.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_1_layout.addWidget(self.additional_table_1)

        # Set up Gage table with 5 columns
        self.additional_table_2 = QTableWidget()
        self.additional_table_2.setColumnCount(5)
        self.additional_table_2.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "JKAM Space Correct", "Summary Statistics"
        ])
        self.additional_table_2.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_2.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_2_layout.addWidget(self.additional_table_2)

        # File button in JKAM table tab
        self.add_file_button_table = QPushButton("Add Files")
        self.add_file_button_table.clicked.connect(self.add_files)
        self.table_layout.addWidget(self.add_file_button_table)

        # Setup the 4 figures
        self.figures = [Figure(), Figure(), Figure(), Figure()]
        self.canvases = [FigureCanvas(fig) for fig in self.figures]

        # Chart layout: 2x2
        # Top-left => JKAM (index=0)
        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        # Top-right => FPGA (index=1)
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        # Bottom-left => FFT (index=2)
        self.chart_layout.addWidget(self.canvases[2], 1, 0)
        # Bottom-right => Gage (index=3)
        self.chart_layout.addWidget(self.canvases[3], 1, 1)

        # File button in charts tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # Initialize plots
        self.initialize_plot(0, "Cumulative Accepted Files 1 (JKAM)")
        self.initialize_plot(1, "Cumulative Accepted Files 2 (Bin/FPGA)")
        self.initialize_fft_plot()
        self.initialize_plot(3, "Cumulative Accepted Files 3 (GageScope)")

        # Handlers
        self.jkam_h5_file_handler = JkamH5FileHandler(self)
        self.bin_handler = BinFileHandler(self)
        self.gage_h5_file_handler = GageScopeH5FileHandler(self)

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

            # Safely handle recognized vs. unrecognized file types
            if file_extension == ".h5":
                # Check if "jkam" or "gage" is in the name
                fname_lower = os.path.basename(file).lower()
                if "jkam" in fname_lower:
                    self.jkam_h5_file_handler.process_file(file)
                elif "gage" in fname_lower:
                    self.gage_h5_file_handler.process_file(file)
                else:
                    print(f"Unsupported .h5 file (not recognized as JKAM or GageScope). Skipping: {file}")
            elif file_extension == ".bin":
                # FPGA
                self.bin_handler.process_file(file)
            else:
                # If it's a "photon timer" or any other extension, skip with no crash
                print(f"Unsupported file type: {file_extension}, skipping file: {file}")

###############################################################################
#                               Main Run                                      #
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
