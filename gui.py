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

        # For the JKAM chart (top-left)
        self.cumulative_data = []

        # For the FFT chart (now in a separate tab)
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

        # Compute average time gap
        if self.shots_num == 0:
            self.start_time = file_ctime
        else:
            curr_time = file_ctime
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

        # Update the JKAM cumulative plot
        self.update_cumulative_plot()
        # Update the FFT plot (now in a separate tab)
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
        Plot the cumulative acceptance of bin (FPGA) files in the top-right chart.
        Points are green if JKAM exists & accepted, red if JKAM doesn't exist or not accepted.
        """
        fig = self.gui.figures[1]
        fig.clear()
        ax = fig.add_subplot(111)

        x_vals = np.arange(len(self.cumulative_data))
        for i in range(len(self.cumulative_data)):
            ax.plot(x_vals[i], self.cumulative_data[i], marker="o", color=self.color_array[i])

        # Connect the points with a line for visual clarity
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
        """
        Bottom-right chart for GageScope acceptance
        """
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
    Handles Red Pitaya .txt files with the provided acceptance logic:
    -----------------------------------------------------------
    1) We'll parse each .txt file using np.loadtxt(...).
    2) We'll check time_temp from JKAM for that shot index.
    3) We'll use the snippet logic for space_correct and acceptance:
    
       space_correct = True
       # check neighbors in jkam_creation_time_array for shot_num-1, shot_num+1
       # check if np.min(abs(rp_creation_time_array - time_temp)) <= 0.3*avg_time_gap
       # if pass => accepted, else => rejected
    -----------------------------------------------------------
    We'll mimic the dynamic approach used by FPGA & Gage:
    - Each .txt file is one "shot_num" for Red Pitaya.
    - Then we re-run acceptance for all shots from scratch.
    """
    def __init__(self, gui):
        self.gui = gui
        self.rp_files = []            # Each file name
        self.rp_times_list = []       # Each entry is an array of times from the file
        self.cumulative_data = []     # For the chart
        self.color_array = []         # 'r' or 'g'
        self.mask_valid_data_rp = []  # True/False acceptance
        self.jkam_rp_matchlist = []   # If accepted, index in rp_times_list

    def process_file(self, file):
        """
        Each new .txt file is a new shot_num for Red Pitaya.
        We'll parse the file, store the times, then re-run acceptance.
        """
        if not os.path.exists(file):
            print(f"File does not exist: {file}")
            return

        # Load data from the .txt file
        try:
            filename_phase = np.loadtxt(file, dtype=float, delimiter=',')
        except Exception as e:
            print(f"Failed to load Red Pitaya file {file}: {e}")
            return

        # Suppose the first column is the "rp_creation_time_array"
        if len(filename_phase.shape) == 1:
            # If it's only one row of data, reshape
            filename_phase = filename_phase.reshape(1, -1)
        rp_creation_time_array = filename_phase[:, 0]  # all time values

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

        # Summaries
        summary_text = (
            f"<b>RP Times Count:</b> {len(rp_creation_time_array)}"
        )
        self.gui.additional_table_3.setItem(row_position, 4, QTableWidgetItem(summary_text))

        # Update the chart
        self.update_chart_rp()

    def rerun_acceptance_rp(self):
        """
        Re-check acceptance for ALL Red Pitaya shots from scratch,
        using the provided logic snippet that references jkam.
        """
        num_shots = len(self.rp_files)

        # We'll create arrays of length num_shots
        self.mask_valid_data_rp = [False]*num_shots
        self.jkam_rp_matchlist = [-1]*num_shots
        self.color_array = ["r"]*num_shots
        self.cumulative_data = []

        # We'll fetch JKAM data
        jkam_ctimes = self.gui.jkam_h5_file_handler.jkam_creation_time_array
        jkam_space_dict = self.gui.jkam_h5_file_handler.shots_dict
        jkam_time_temp_dict = self.gui.jkam_h5_file_handler.time_temp_dict
        jkam_avg_time_gap = self.gui.jkam_h5_file_handler.avg_time_gap
        total_jkam_shots = len(jkam_ctimes)

        # For each Red Pitaya shot (same "shot_num" as index)
        for shot_num in range(num_shots):
            # If there's no JKAM for this shot_num, remain red & skip
            if shot_num not in jkam_time_temp_dict or shot_num not in jkam_space_dict:
                self.mask_valid_data_rp[shot_num] = False
                self.color_array[shot_num] = "r"
                self.jkam_rp_matchlist[shot_num] = -1
                continue

            # We do have JKAM
            time_temp = jkam_time_temp_dict[shot_num]
            jkam_space_correct = jkam_space_dict[shot_num]

            # from the snippet:
            space_correct = True
            # If shot_num>0, check jkam neighbor -1
            if shot_num > 0 and shot_num < total_jkam_shots:
                if abs(time_temp - jkam_ctimes[shot_num - 1] - jkam_avg_time_gap) > 0.3*jkam_avg_time_gap:
                    space_correct = False
            # If shot_num<(num_shots-1), check jkam neighbor +1
            # but carefully ensure shot_num+1 < total_jkam_shots
            if shot_num < (total_jkam_shots - 1):
                if abs(-time_temp + jkam_ctimes[shot_num+1] - jkam_avg_time_gap) > 0.3*jkam_avg_time_gap:
                    space_correct = False

            # Now check the Red Pitaya times for that shot
            rp_creation_time_array = self.rp_times_list[shot_num]
            rp_index_list = np.arange(len(rp_creation_time_array))

            # from snippet:
            # if (np.min(np.abs(rp_creation_time_array - time_temp)) <= 0.3*avg_time_gap) & space_correct
            # We'll treat avg_time_gap as jkam_avg_time_gap
            if jkam_space_correct and space_correct:
                min_diff = np.min(np.abs(rp_creation_time_array - time_temp))
                if min_diff <= 0.3*jkam_avg_time_gap:
                    self.mask_valid_data_rp[shot_num] = True
                    idx = np.argmin(np.abs(rp_creation_time_array - time_temp))
                    self.jkam_rp_matchlist[shot_num] = rp_index_list[idx]
                    self.color_array[shot_num] = "g"
                else:
                    print(f"error at {shot_num}")
                    self.mask_valid_data_rp[shot_num] = False
                    self.color_array[shot_num] = "r"
                    self.jkam_rp_matchlist[shot_num] = -1
            else:
                # either jkam wasn't space_correct or neighbor check failed
                print(f"error at {shot_num}")
                self.mask_valid_data_rp[shot_num] = False
                self.color_array[shot_num] = "r"
                self.jkam_rp_matchlist[shot_num] = -1

            # Build the cumulative_data array
            if self.mask_valid_data_rp[shot_num]:
                new_val = (self.cumulative_data[-1] + 1) if self.cumulative_data else 1
            else:
                new_val = 0
            self.cumulative_data.append(new_val)

    def update_chart_rp(self):
        """
        Bottom-left chart for Red Pitaya acceptance.
        """
        fig = self.gui.figures[2]  # we placed Red Pitaya in bottom-left now
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
#                           Main GUI                                          #
###############################################################################
class FileProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Processor GUI")
        # Make the window wide enough to display 2x2 charts plus extra tabs
        self.setGeometry(100, 100, 1600, 900)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.layout = QVBoxLayout(self.central_widget)

        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

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
        # JKAM table (4 columns)
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
        # FPGA table (5 columns)
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
        # GageScope table (5 columns)
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
        # Red Pitaya table (5 columns)
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
        # Figures layout in the "Charts" tab: a 2x2 grid for (0, 1, 2, 3),
        # and figure[4] in the "FFT Graph" tab
        ############################################################################
        self.figures = [Figure() for _ in range(5)]
        self.canvases = [FigureCanvas(fig) for fig in self.figures]

        # Place 4 of them in a 2x2 layout in the chart tab
        # top-left => JKAM (figures[0])
        self.chart_layout.addWidget(self.canvases[0], 0, 0)
        # top-right => FPGA (figures[1])
        self.chart_layout.addWidget(self.canvases[1], 0, 1)
        # bottom-left => Red Pitaya (figures[2])
        self.chart_layout.addWidget(self.canvases[2], 1, 0)
        # bottom-right => GageScope (figures[3])
        self.chart_layout.addWidget(self.canvases[3], 1, 1)

        # File button in Charts tab
        self.add_file_button_charts = QPushButton("Add Files")
        self.add_file_button_charts.clicked.connect(self.add_files)
        self.chart_layout.addWidget(self.add_file_button_charts, 2, 0, 1, 2)

        # Figure[4] is the FFT chart, placed in the separate "FFT Graph" tab
        self.fft_tab_layout.addWidget(self.canvases[4])

        # Initialize each chart
        self.initialize_plot(0, "Cumulative Accepted Files 1 (JKAM)")
        self.initialize_plot(1, "Cumulative Accepted Files 2 (Bin/FPGA)")
        self.initialize_plot(2, "Cumulative Accepted Files (Red Pitaya)")
        self.initialize_plot(3, "Cumulative Accepted Files 3 (GageScope)")
        self.initialize_fft_plot(4)

        # Handlers
        self.jkam_h5_file_handler = JkamH5FileHandler(self)
        self.bin_handler = BinFileHandler(self)
        self.gage_h5_file_handler = GageScopeH5FileHandler(self)
        self.redpitaya_handler = RedPitayaFileHandler(self)

    def initialize_plot(self, index, title_str):
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [], marker="o")
        ax.set_title(title_str)
        ax.set_xlabel("Shot Number")
        ax.set_ylabel("Cumulative Value")
        self.canvases[index].draw()

    def initialize_fft_plot(self, index):
        """
        Setup the figure used for FFT in the new 'FFT Graph' tab
        """
        ax = self.figures[index].add_subplot(111)
        ax.plot([], [])
        ax.set_title("FFT of the Signal")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.canvases[index].draw()

    def add_files(self):
        """
        Common file-adding function used by both the table and charts tab buttons.
        """
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*.*)")
        if not files:
            return

        for file in files:
            file_extension = os.path.splitext(file)[-1].lower()
            fname_lower = os.path.basename(file).lower()

            # JKAM or Gage
            if file_extension == ".h5":
                if "jkam" in fname_lower:
                    self.jkam_h5_file_handler.process_file(file)
                elif "gage" in fname_lower:
                    self.gage_h5_file_handler.process_file(file)
                else:
                    print(f"Unsupported .h5 file (not recognized as JKAM or GageScope). Skipping: {file}")

            # FPGA
            elif file_extension == ".bin":
                self.bin_handler.process_file(file)

            # Red Pitaya .txt
            elif file_extension == ".txt":
                self.redpitaya_handler.process_file(file)

            else:
                # If it's something else (e.g. photon timer extension, etc.), skip
                print(f"Unsupported file extension: {file_extension}, skipping file: {file}")


###############################################################################
#                               Main Run                                      #
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FileProcessorGUI()
    main_window.show()
    sys.exit(app.exec_())
