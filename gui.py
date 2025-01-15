import sys
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QWidget, QTabWidget, QGridLayout, QHeaderView,
    QLabel, QHBoxLayout, QLineEdit, QDockWidget, QCheckBox
)
from PyQt5.QtCore import QTimer, Qt
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

        # For the JKAM chart (top-left)
        self.cumulative_data = []
        self.highest_count = 0  # Track highest count reached

        # For the FFT chart (in a separate tab)
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

        if file in self.jkam_files:
            # Already processed
            return

        self.jkam_files.append(file)
        self.jkam_creation_time_array.append(file_ctime)

        space_correct = True
        time_temp = file_ctime

        # Compute average time gap
        if self.shots_num == 0:
            self.start_time = file_ctime
        else:
            curr_time = file_ctime
            self.avg_time_gap = abs((curr_time - self.start_time) / self.shots_num)
            prev_time_temp = self.jkam_creation_time_array[self.shots_num - 1]

            # Check if this shot's time is near the expected spacing
            expected_time_temp = prev_time_temp + self.avg_time_gap
            normalized_difference = abs(time_temp - expected_time_temp) / self.avg_time_gap
            if (normalized_difference > .3):
                space_correct = False

        # Store space_correct & time_temp for this shot index
        self.shots_dict[self.shots_num] = space_correct
        self.time_temp_dict[self.shots_num] = time_temp

        # Modified cumulative data calculation
        if space_correct:
            if not self.cumulative_data or self.cumulative_data[-1] == 0:
                # If first point or previous was failure, start from highest + 1
                new_val = self.highest_count + 1
            else:
                new_val = self.cumulative_data[-1] + 1
            self.highest_count = max(self.highest_count, new_val)
        else:
            new_val = 0
        self.cumulative_data.append(new_val)

        self.shots_num += 1

        # For FFT example, store the creation time
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

        # Update the JKAM cumulative plot
        self.update_cumulative_plot()
        # Update the FFT plot (in a separate tab)
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
        Moved the FFT display to its own tab (figure[4]).
        We'll only run if we have at least 2 data points.
        """
        if len(self.all_datapoints) < 2:
            return

        fig = self.gui.figures[4]  # figure index 4 => FFT
        fig.clear()
        ax = fig.add_subplot(111)

        # Perform FFT of creation times (just a demonstration!)
        fft_result = np.fft.fft(self.all_datapoints)
        freqs = np.fft.fftfreq(len(self.all_datapoints))

        ax.plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.gui.canvases[4].draw()


###############################################################################
#                      FPGA / Bin Handler                                     #
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
        self.color_array = []
        self.cumulative_data = []
        self.highest_count = 0  # Track highest count reached

        # Lock in once accepted (so older shots don't flip red)
        self.final_accepted = []

        # Tracking
        self.start_time = None
        self.avg_time_gap = 0

    def process_file(self, file):
        """
        Each time we get a new bin file, let's store it, then we will
        re-run acceptance logic for the entire bin-file list from scratch.
        """
        if file in self.bin_files:
            return

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

        # Finally, update the table with info for this newly added file
        new_shot_index = len(self.fpga_creation_time_array) - 1
        data_valid = False
        jkam_space_correct_str = "None"

        if 0 <= new_shot_index < len(self.mask_valid_data):
            data_valid = self.mask_valid_data[new_shot_index]
        # If JKAM existed for that shot, show whether it was space-correct
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        if new_shot_index in jkam_space_dict:
            jkam_space_correct_str = str(jkam_space_dict[new_shot_index])

        # Add row to Additional Table 1 (FPGA)
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
        # >>>> FIX: Reset highest_count here so it doesn't inflate on repeated runs <<<<
        self.highest_count = 0

        num_shots = len(self.fpga_creation_time_array)

        # Ensure final_accepted is long enough to hold all shots
        if len(self.final_accepted) < num_shots:
            self.final_accepted += [False] * (num_shots - len(self.final_accepted))

        if num_shots <= 1:
            self.avg_time_gap = 0
        else:
            total_span = abs(self.fpga_creation_time_array[-1] - self.fpga_creation_time_array[0])
            self.avg_time_gap = abs(total_span / (num_shots - 1))

        self.mask_valid_data = np.zeros(num_shots, dtype=bool)
        self.jkam_fpga_matchlist = np.full(num_shots, -1, dtype=int)
        self.color_array = ["r"] * num_shots

        # Retrieve JKAM dictionaries
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict

        fpga_ctimes = np.array(self.fpga_creation_time_array)
        fpga_index_list = np.arange(num_shots)

        for shot_num in range(num_shots):
            # If already accepted, skip re-check
            if self.final_accepted[shot_num]:
                self.mask_valid_data[shot_num] = True
                self.color_array[shot_num] = "g"
                continue

            if shot_num in jkam_time_temp_dict and shot_num in jkam_space_dict:
                jkam_time = jkam_time_temp_dict[shot_num]
                space_correct = jkam_space_dict[shot_num]

                if self.avg_time_gap == 0:
                    if space_correct:
                        self.mask_valid_data[shot_num] = True
                        self.color_array[shot_num] = "g"
                        self.jkam_fpga_matchlist[shot_num] = shot_num
                        self.final_accepted[shot_num] = True
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.color_array[shot_num] = "r"
                else:
                    time_diffs = np.abs(fpga_ctimes - jkam_time)
                    min_diff = np.min(time_diffs)

                    if (min_diff <= 0.3 * self.avg_time_gap) and space_correct:
                        self.mask_valid_data[shot_num] = True
                        closest_idx = np.argmin(time_diffs)
                        self.jkam_fpga_matchlist[shot_num] = fpga_index_list[closest_idx]
                        self.color_array[shot_num] = "g"
                        self.final_accepted[shot_num] = True
                    else:
                        self.mask_valid_data[shot_num] = False
                        self.color_array[shot_num] = "r"
                        print(f"FPGA error at shot {shot_num}")
            else:
                # If no JKAM data, remain red & mask = False
                self.mask_valid_data[shot_num] = False
                self.jkam_fpga_matchlist[shot_num] = -1

        # Modified cumulative data calculation
        self.cumulative_data = []
        current_count = 0
        
        for shot_num in range(num_shots):
            if self.mask_valid_data[shot_num]:
                if not self.cumulative_data or self.cumulative_data[-1] == 0:
                    # If first point or previous was failure, start from highest + 1
                    current_count = self.highest_count + 1
                else:
                    current_count += 1
                self.highest_count = max(self.highest_count, current_count)
                self.cumulative_data.append(current_count)
            else:
                self.cumulative_data.append(0)

    def update_chart_2(self):
        fig = self.gui.figures[1]
        fig.clear()
        ax = fig.add_subplot(111)

        x_vals = np.arange(len(self.cumulative_data))
        for i in range(len(self.cumulative_data)):
            ax.plot(x_vals[i], self.cumulative_data[i], marker="o", color=self.color_array[i])

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
    Handles GageScope .h5 files with the same acceptance logic as Bin/FPGA,
    but with "sawtooth" approach for the cumulative chart.
    """
    def __init__(self, gui):
        self.gui = gui

        self.gage_files = []
        self.gage_creation_time_array = []

        # Acceptance logic arrays
        self.mask_valid_data = []
        self.jkam_gage_matchlist = []
        self.color_array = []
        self.cumulative_data = []

        # Once a shot is accepted, lock it in
        self.final_accepted = []

        # Tracking
        self.start_time = None
        self.avg_time_gap = 0

    def process_file(self, file):
        if file in self.gage_files:
            return

        try:
            file_ctime = os.path.getctime(file)
        except Exception as e:
            print(f"Error accessing file time for {file}: {e}")
            return

        self.gage_files.append(file)
        self.gage_creation_time_array.append(file_ctime)

        if len(self.gage_creation_time_array) == 1:
            self.start_time = file_ctime

        self.rerun_acceptance_gage()

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

        # Make sure final_accepted is big enough
        if len(self.final_accepted) < num_shots:
            self.final_accepted += [False] * (num_shots - len(self.final_accepted))

        if num_shots <= 1:
            self.avg_time_gap = 0
        else:
            total_span = (self.gage_creation_time_array[-1]
                          - self.gage_creation_time_array[0])
            self.avg_time_gap = total_span / (num_shots - 1)

        self.mask_valid_data = np.zeros(num_shots, dtype=bool)
        self.jkam_gage_matchlist = np.full(num_shots, -1, dtype=int)
        self.color_array = ["r"] * num_shots

        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict

        gage_ctimes = np.array(self.gage_creation_time_array)
        gage_index_list = np.arange(num_shots)

        for shot_num in range(num_shots):
            # If we've already locked acceptance, skip re-check
            if self.final_accepted[shot_num]:
                self.mask_valid_data[shot_num] = True
                self.color_array[shot_num] = "g"
                continue

            if shot_num in jkam_time_temp_dict and shot_num in jkam_space_dict:
                jkam_time = jkam_time_temp_dict[shot_num]
                space_correct = jkam_space_dict[shot_num]

                if space_correct:
                    if self.avg_time_gap == 0:
                        self.mask_valid_data[shot_num] = True
                        self.color_array[shot_num] = "g"
                        self.jkam_gage_matchlist[shot_num] = shot_num
                        self.final_accepted[shot_num] = True
                    else:
                        time_diffs = np.abs(gage_ctimes - jkam_time)
                        min_diff = np.min(time_diffs)

                        if (min_diff <= 0.3 * self.avg_time_gap):
                            self.mask_valid_data[shot_num] = True
                            closest_idx = np.argmin(time_diffs)
                            self.jkam_gage_matchlist[shot_num] = gage_index_list[closest_idx]
                            self.color_array[shot_num] = "g"
                            self.final_accepted[shot_num] = True
                        else:
                            self.mask_valid_data[shot_num] = False
                            self.color_array[shot_num] = "r"
                            print(f"Gage error at shot {shot_num}")
                else:
                    self.mask_valid_data[shot_num] = False
                    self.color_array[shot_num] = "r"
            else:
                self.mask_valid_data[shot_num] = False
                self.jkam_gage_matchlist[shot_num] = -1

        # Build cumulative data with modified logic
        self.cumulative_data = []
        last_success_count = 0
        highest_count = 0
        
        for shot_num in range(num_shots):
            if self.mask_valid_data[shot_num]:
                if not self.cumulative_data or self.cumulative_data[-1] == 0:
                    # If previous was a failure or this is first point
                    last_success_count = highest_count + 1
                else:
                    last_success_count += 1
                highest_count = max(highest_count, last_success_count)
                self.cumulative_data.append(last_success_count)
            else:
                self.cumulative_data.append(0)

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
#                     Red Pitaya Handler for .txt files                       #
###############################################################################
class RedPitayaFileHandler:
    """
    Handles Red Pitaya .txt files with the provided acceptance logic.
    If a file is empty or blank, we do not crash the GUI. We just mark it as
    an error on the chart and move on.
    """
    def __init__(self, gui):
        self.gui = gui
        self.rp_files = []            # Each file name
        self.rp_times_list = []       # Each entry is an array of times from the file

        self.mask_valid_data_rp = []  # True/False acceptance
        self.jkam_rp_matchlist = []   # If accepted, index in rp_times_list
        self.color_array = []
        self.cumulative_data = []

        # Once accepted, lock it
        self.final_accepted = []

    def process_file(self, file):
        """
        Each new .txt file is a new shot_num for Red Pitaya.
        We'll parse the file, store the times, then re-run acceptance.
        If the file is empty, we skip the data load and label it as a failure.
        """
        if file in self.rp_files:
            return

        if not os.path.exists(file):
            print(f"File does not exist: {file}")
            return

        # Load data
        try:
            filename_phase = np.loadtxt(file, dtype=float, delimiter=',')
        except Exception as e:
            # Could be empty or any parse error
            print(f"Failed to load Red Pitaya file {file}: {e}")
            self.rp_files.append(file)
            self.rp_times_list.append(None)  # No data
            self.rerun_acceptance_rp()
            return

        if filename_phase.size == 0:
            print(f"Error: Red Pitaya file is empty: {file}")
            self.rp_files.append(file)
            self.rp_times_list.append(None)
            self.rerun_acceptance_rp()
            return

        # If there's only one column, ensure shape fits
        if len(filename_phase.shape) == 1:
            filename_phase = filename_phase.reshape(1, -1)

        # Suppose the first column is the "rp_creation_time_array"
        rp_creation_time_array = filename_phase[:, 0]

        self.rp_files.append(file)
        self.rp_times_list.append(rp_creation_time_array)

        # Rerun acceptance for ALL red pitaya shots
        self.rerun_acceptance_rp()

        # Add a row to Additional Table 3
        new_shot_index = len(self.rp_files) - 1
        data_valid = False
        if (0 <= new_shot_index < len(self.mask_valid_data_rp)):
            data_valid = self.mask_valid_data_rp[new_shot_index]

        # Check if JKAM was space_correct
        jkam_space_correct_str = "None"
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        if new_shot_index in jkam_space_dict:
            jkam_space_correct_str = str(jkam_space_dict[new_shot_index])

        row_position = self.gui.additional_table_3.rowCount()
        self.gui.additional_table_3.insertRow(row_position)
        self.gui.additional_table_3.setItem(row_position, 0, QTableWidgetItem(str(new_shot_index)))
        self.gui.additional_table_3.setItem(row_position, 1, QTableWidgetItem(file))
        self.gui.additional_table_3.setItem(row_position, 2, QTableWidgetItem(str(data_valid)))
        self.gui.additional_table_3.setItem(row_position, 3, QTableWidgetItem(jkam_space_correct_str))

        info_str = ("No Data" if (rp_creation_time_array is None or len(rp_creation_time_array) == 0)
                    else f"RP Times Count: {len(rp_creation_time_array)}")
        self.gui.additional_table_3.setItem(row_position, 4, QTableWidgetItem(info_str))

        # Update the chart
        self.update_chart_rp()

    def rerun_acceptance_rp(self):
        """
        Re-check acceptance for ALL Red Pitaya shots from scratch,
        using the snippet logic that references JKAM.
        If a shot has None data, it is automatically red/failure.
        """
        num_shots = len(self.rp_files)

        # Ensure final_accepted is long enough
        if len(self.final_accepted) < num_shots:
            self.final_accepted += [False]*(num_shots - len(self.final_accepted))

        self.mask_valid_data_rp = [False]*num_shots
        self.jkam_rp_matchlist = [-1]*num_shots
        self.color_array = ["r"]*num_shots
        self.cumulative_data = []
        highest_count = 0
        last_success_count = 0

        # We'll fetch JKAM data
        jkam_ctimes = self.gui.jkam_h5_file_handler.jkam_creation_time_array
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict
        jkam_avg_time_gap = self.gui.jkam_h5_file_handler.avg_time_gap

        for shot_num in range(num_shots):
            # If we've already locked acceptance, keep it
            if self.final_accepted[shot_num]:
                self.mask_valid_data_rp[shot_num] = True
                self.color_array[shot_num] = "g"
                if not self.cumulative_data or self.cumulative_data[-1] == 0:
                    last_success_count = highest_count + 1
                else:
                    last_success_count += 1
                highest_count = max(highest_count, last_success_count)
                self.cumulative_data.append(last_success_count)
                continue

            rp_creation_time_array = self.rp_times_list[shot_num]
            if rp_creation_time_array is None or len(rp_creation_time_array) == 0:
                # It's blank or invalid
                self.mask_valid_data_rp[shot_num] = False
                self.color_array[shot_num] = "r"
                self.jkam_rp_matchlist[shot_num] = -1
                self.cumulative_data.append(0)
                continue

            if (shot_num not in jkam_time_temp_dict) or (shot_num not in jkam_space_dict):
                # no JKAM => fail
                self.mask_valid_data_rp[shot_num] = False
                self.color_array[shot_num] = "r"
                self.jkam_rp_matchlist[shot_num] = -1
                self.cumulative_data.append(0)
                continue

            # We do have JKAM
            time_temp = jkam_time_temp_dict[shot_num]
            jkam_space_correct = jkam_space_dict[shot_num]

            if jkam_space_correct:
                # Check if at least one data point in RP is within 0.3 * jkam_avg_time_gap of JKAM time
                rp_index_list = np.arange(len(rp_creation_time_array))
                min_diff = np.min(np.abs(rp_creation_time_array - time_temp))
                if min_diff <= 0.3*jkam_avg_time_gap:
                    self.mask_valid_data_rp[shot_num] = True
                    idx = np.argmin(np.abs(rp_creation_time_array - time_temp))
                    self.jkam_rp_matchlist[shot_num] = rp_index_list[idx]
                    self.color_array[shot_num] = "g"
                    self.final_accepted[shot_num] = True
                    if not self.cumulative_data or self.cumulative_data[-1] == 0:
                        last_success_count = highest_count + 1
                    else:
                        last_success_count += 1
                    highest_count = max(highest_count, last_success_count)
                    self.cumulative_data.append(last_success_count)
                else:
                    print(f"error at {shot_num}")
                    self.mask_valid_data_rp[shot_num] = False
                    self.color_array[shot_num] = "r"
                    self.jkam_rp_matchlist[shot_num] = -1
                    self.cumulative_data.append(0)
            else:
                print(f"error at {shot_num}")
                self.mask_valid_data_rp[shot_num] = False
                self.color_array[shot_num] = "r"
                self.jkam_rp_matchlist[shot_num] = -1
                self.cumulative_data.append(0)

    def update_chart_rp(self):
        """
        Bottom-left chart for Red Pitaya acceptance.
        """
        fig = self.gui.figures[2]  # we placed Red Pitaya in bottom-left
        fig.clear()
        ax = fig.add_subplot(111)

        x_vals = np.arange(len(self.cumulative_data))
        for i in range(len(self.cumulative_data)):
            ax.plot(x_vals[i], self.cumulative_data[i], marker="o", color=self.color_array[i])
        ax.plot(x_vals, self.cumulative_data, linestyle="-", alpha=0.3)

        ax.set_title("Cumulative Accepted Files (Red Pitaya)")
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.gui.canvases[2].draw()


###############################################################################
#                           Main GUI (Modified)                               #
###############################################################################
class FileProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Processor GUI")
        self.setGeometry(100, 100, 1600, 900)

        # A flag to indicate if the user has accepted the inputs
        self.inputs_accepted = False

        # --------------------------------------------------------------
        # IMPORTANT FIX: Initialize the file handlers so we don't crash
        self.jkam_h5_file_handler = JkamH5FileHandler(self)
        self.gage_h5_file_handler = GageScopeH5FileHandler(self)
        self.bin_handler = BinFileHandler(self)
        self.redpitaya_handler = RedPitayaFileHandler(self)
        # --------------------------------------------------------------

        # Create a central widget with a horizontal layout,
        # so we can have the new feature-options panel on the left
        # and the existing tabs/controls on the right.
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_hlayout = QHBoxLayout(self.central_widget)

        # -------------------- New Left Panel with Feature Options --------------------
        self.leftDock = QDockWidget("Feature Options", self)
        self.leftDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.leftDock)

        self.feature_options_widget = QWidget()
        self.feature_options_layout = QVBoxLayout(self.feature_options_widget)

        # 1) Booleans
        self.time_me_checkbox = QCheckBox("time_me")
        self.plot_tenth_shot_checkbox = QCheckBox("plot_tenth_shot")
        self.feature_options_layout.addWidget(self.time_me_checkbox)
        self.feature_options_layout.addWidget(self.plot_tenth_shot_checkbox)

        # 2) Numeric inputs
        self.het_freq_label = QLabel("het_freq (MHz):")
        self.het_freq_input = QLineEdit()
        self.feature_options_layout.addWidget(self.het_freq_label)
        self.feature_options_layout.addWidget(self.het_freq_input)

        self.dds_freq_label = QLabel("dds_freq:")
        self.dds_freq_input = QLineEdit()
        self.feature_options_layout.addWidget(self.dds_freq_label)
        self.feature_options_layout.addWidget(self.dds_freq_input)

        self.samp_freq_label = QLabel("samp_freq (MHz):")
        self.samp_freq_input = QLineEdit()
        self.feature_options_layout.addWidget(self.samp_freq_label)
        self.feature_options_layout.addWidget(self.samp_freq_input)

        self.averaging_time_label = QLabel("averaging_time (us):")
        self.averaging_time_input = QLineEdit()
        self.feature_options_layout.addWidget(self.averaging_time_label)
        self.feature_options_layout.addWidget(self.averaging_time_input)

        self.step_time_label = QLabel("step_time (us):")
        self.step_time_input = QLineEdit()
        self.feature_options_layout.addWidget(self.step_time_label)
        self.feature_options_layout.addWidget(self.step_time_input)

        self.filter_time_label = QLabel("filter_time (us):")
        self.filter_time_input = QLineEdit()
        self.feature_options_layout.addWidget(self.filter_time_label)
        self.feature_options_layout.addWidget(self.filter_time_input)

        self.voltage_conversion_label = QLabel("voltage_conversion (mV):")
        self.voltage_conversion_input = QLineEdit()
        self.feature_options_layout.addWidget(self.voltage_conversion_label)
        self.feature_options_layout.addWidget(self.voltage_conversion_input)

        self.kappa_label = QLabel("kappa (MHz):")
        self.kappa_input = QLineEdit()
        self.feature_options_layout.addWidget(self.kappa_label)
        self.feature_options_layout.addWidget(self.kappa_input)

        self.LO_power_label = QLabel("LO_power (uW):")
        self.LO_power_input = QLineEdit()
        self.feature_options_layout.addWidget(self.LO_power_label)
        self.feature_options_layout.addWidget(self.LO_power_input)

        self.PHOTON_ENERGY_label = QLabel("PHOTON_ENERGY:")
        self.PHOTON_ENERGY_input = QLineEdit()
        self.feature_options_layout.addWidget(self.PHOTON_ENERGY_label)
        self.feature_options_layout.addWidget(self.PHOTON_ENERGY_input)

        self.LO_rate_label = QLabel("LO_rate (count/us):")
        self.LO_rate_input = QLineEdit()
        self.feature_options_layout.addWidget(self.LO_rate_label)
        self.feature_options_layout.addWidget(self.LO_rate_input)

        self.photonrate_conversion_label = QLabel("photonrate_conversion (count/us):")
        self.photonrate_conversion_input = QLineEdit()
        self.feature_options_layout.addWidget(self.photonrate_conversion_label)
        self.feature_options_layout.addWidget(self.photonrate_conversion_input)

        # Accept Button
        self.accept_button = QPushButton("Accept Inputs")
        self.accept_button.clicked.connect(self.accept_inputs)
        self.feature_options_layout.addWidget(self.accept_button)

        # Warning label to put in inputs or it won't start
        self.inputs_status_label = QLabel(
            "PLEASE GIVE INPUTS AND CLICK \"Accept Inputs \" BUTTON TO START PROCESSING FILES!"
        )
        self.feature_options_layout.addWidget(self.inputs_status_label)

        self.feature_options_layout.addStretch()
        self.feature_options_widget.setLayout(self.feature_options_layout)
        self.leftDock.setWidget(self.feature_options_widget)

        # -------------------- Right-Side Tabs --------------------
        self.right_side_widget = QWidget()
        self.right_side_layout = QVBoxLayout(self.right_side_widget)
        self.main_hlayout.addWidget(self.right_side_widget)

        self.tabs = QTabWidget()
        self.right_side_layout.addWidget(self.tabs)

        # 1) Chart tab
        self.chart_tab = QWidget()
        self.chart_layout = QGridLayout(self.chart_tab)
        self.tabs.addTab(self.chart_tab, "Charts")

        # 2) JKAM table tab
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.tabs.addTab(self.table_tab, "JKAM Data Table")

        # 3) FPGA table tab
        self.additional_table_tab_1 = QWidget()
        self.additional_table_tab_1_layout = QVBoxLayout(self.additional_table_tab_1)
        self.tabs.addTab(self.additional_table_tab_1, "Additional Table 1 (FPGA)")

        # 4) GageScope table tab
        self.additional_table_tab_2 = QWidget()
        self.additional_table_tab_2_layout = QVBoxLayout(self.additional_table_tab_2)
        self.tabs.addTab(self.additional_table_tab_2, "Additional Table 2 (GageScope)")

        # 5) Red Pitaya table tab
        self.additional_table_tab_3 = QWidget()
        self.additional_table_tab_3_layout = QVBoxLayout(self.additional_table_tab_3)
        self.tabs.addTab(self.additional_table_tab_3, "Additional Table 3 (Red Pitaya)")

        # 6) FFT Graph tab
        self.fft_tab = QWidget()
        self.fft_tab_layout = QVBoxLayout(self.fft_tab)
        self.tabs.addTab(self.fft_tab, "FFT Graph")

        ############################################################################
        # JKAM table
        ############################################################################
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Shot Number", "File Name", "Accepted", "Summary Statistics"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout.addWidget(self.table)

        # Button in JKAM table tab
        self.add_file_button_table = QPushButton("Add Files")
        self.add_file_button_table.clicked.connect(self.add_files)
        self.table_layout.addWidget(self.add_file_button_table)

        ############################################################################
        # FPGA table
        ############################################################################
        self.additional_table_1 = QTableWidget()
        self.additional_table_1.setColumnCount(5)
        self.additional_table_1.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "JKAM Space Correct", "Summary Statistics"
        ])
        self.additional_table_1.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_1.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_1_layout.addWidget(self.additional_table_1)

        ############################################################################
        # GageScope table
        ############################################################################
        self.additional_table_2 = QTableWidget()
        self.additional_table_2.setColumnCount(5)
        self.additional_table_2.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "JKAM Space Correct", "Summary Statistics"
        ])
        self.additional_table_2.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_2.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_2_layout.addWidget(self.additional_table_2)

        ############################################################################
        # Red Pitaya table
        ############################################################################
        self.additional_table_3 = QTableWidget()
        self.additional_table_3.setColumnCount(5)
        self.additional_table_3.setHorizontalHeaderLabels([
            "Shot Number", "File Name", "Accepted", "JKAM Space Correct", "Summary Statistics"
        ])
        self.additional_table_3.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.additional_table_3.horizontalHeader().setStretchLastSection(True)
        self.additional_table_tab_3_layout.addWidget(self.additional_table_3)

        ############################################################################
        # Setup the 5 figures
        ############################################################################
        self.figures = [Figure() for _ in range(5)]
        self.canvases = [FigureCanvas(fig) for fig in self.figures]

        # Place 4 of them in a 2x2 layout in the chart tab
        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        self.chart_layout.addWidget(self.canvases[2], 1, 0)
        self.chart_layout.addWidget(self.canvases[3], 1, 1)

        # File button in Charts tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # figure[4] is the FFT chart
        self.fft_tab_layout.addWidget(self.canvases[4])

        # Initialize each chart
        self.initialize_plot(0, "Cumulative Accepted Files 1 (JKAM)")
        self.initialize_plot(1, "Cumulative Accepted Files 2 (Bin/FPGA)")
        self.initialize_plot(2, "Cumulative Accepted Files (Red Pitaya)")
        self.initialize_plot(3, "Cumulative Accepted Files 3 (GageScope)")
        self.initialize_fft_plot(4)

        # ------------------- Streaming Controls -------------------
        self.stream_controls_layout = QHBoxLayout()
        self.stream_dir_label = QLabel("Stream Directory:")
        self.stream_dir_edit = QLineEdit(os.getcwd())
        self.stream_start_button = QPushButton("Start Stream")
        self.stream_stop_button = QPushButton("Stop Stream")

        # Add a status label to indicate streaming state
        self.stream_status_label = QLabel("Not streaming")

        self.stream_controls_layout.addWidget(self.stream_dir_label)
        self.stream_controls_layout.addWidget(self.stream_dir_edit)
        self.stream_controls_layout.addWidget(self.stream_start_button)
        self.stream_controls_layout.addWidget(self.stream_stop_button)
        self.stream_controls_layout.addWidget(self.stream_status_label)
        self.right_side_layout.addLayout(self.stream_controls_layout)

        # Initialize QTimer for streaming
        self.stream_timer = QTimer()
        self.stream_timer.setInterval(2000)
        self.stream_timer.timeout.connect(self.check_for_new_files)

        # Connect start/stop
        self.stream_start_button.clicked.connect(self.start_stream)
        self.stream_stop_button.clicked.connect(self.stop_stream)

        # Keep track of files we've already processed
        self.stream_processed_files = set()

    def accept_inputs(self):
        """
        Check if all relevant inputs are filled in. If yes, set inputs_accepted to True
        and enable the rest of the functionality. Otherwise, print a message.
        """
        # Minimal check: ensure all QLineEdits are non-empty if they are relevant
        required_fields = [
            self.het_freq_input, self.dds_freq_input, self.samp_freq_input,
            self.averaging_time_input, self.step_time_input, self.filter_time_input,
            self.voltage_conversion_input, self.kappa_input, self.LO_power_input,
            self.PHOTON_ENERGY_input, self.LO_rate_input, self.photonrate_conversion_input
        ]
        for field in required_fields:
            if field.text().strip() == "":
                print("Please fill in all inputs before accepting.")
                self.inputs_accepted = False
                return

        # If we got here, all fields are filled
        self.inputs_accepted = True
        self.inputs_status_label.setText("Inputs accepted! You may now use the rest of the GUI!")
        print("Inputs accepted! You may now use the rest of the GUI.")

    def initialize_plot(self, index, title_str):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [], marker="o")
        ax.set_title(title_str)
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.canvases[index].draw()

    def initialize_fft_plot(self, index):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.canvases[index].draw()

    def add_files(self):
        """
        Only works if inputs have been accepted.
        """
        if not self.inputs_accepted:
            print("Please fill in all inputs and click 'Accept Inputs' first.")
            return

        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*.*)")
        if not files:
            return

        for file in files:
            self.process_one_file(file)

    def process_one_file(self, file):
        """
        Helper function for adding/streaming files. 
        We assume inputs_accepted is already True if we get here.
        """
        file_extension = os.path.splitext(file)[-1].lower()
        fname_lower = os.path.basename(file).lower()

        if file_extension == ".h5":
            if "jkam" in fname_lower:
                self.jkam_h5_file_handler.process_file(file)
            elif "gage" in fname_lower:
                self.gage_h5_file_handler.process_file(file)
            else:
                print(f"Unsupported .h5 file (not recognized as JKAM or GageScope). Skipping: {file}")
        elif file_extension == ".bin":
            self.bin_handler.process_file(file)
        elif file_extension == ".txt":
            self.redpitaya_handler.process_file(file)
        else:
            print(f"Unsupported file extension: {file_extension}, skipping file: {file}")

    def start_stream(self):
        """
        Start streaming if inputs are accepted.
        """
        if not self.inputs_accepted:
            print("Please fill in all inputs and click 'Accept Inputs' first.")
            return

        self.stream_processed_files = set()
        self.stream_timer.start()
        self.stream_status_label.setText("Streaming has started!")
        print("Stream started. Monitoring directory:", self.stream_dir_edit.text())

    def stop_stream(self):
        self.stream_timer.stop()
        self.stream_status_label.setText("Not streaming")
        print("Stream stopped.")

    def check_for_new_files(self):
        if not self.inputs_accepted:
            return  # If inputs are not accepted, do nothing

        watch_dir = self.stream_dir_edit.text()
        if not os.path.isdir(watch_dir):
            print(f"Invalid stream directory: {watch_dir}")
            return

        all_files = [os.path.join(watch_dir, f) for f in os.listdir(watch_dir)]
        all_files = [f for f in all_files if os.path.isfile(f)]

        new_files = [f for f in all_files if f not in self.stream_processed_files]
        for nf in new_files:
            self.process_one_file(nf)
            self.stream_processed_files.add(nf)


###############################################################################
#                               Main Run                                      #
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
