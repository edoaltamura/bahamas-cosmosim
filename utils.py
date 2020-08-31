import os
import numpy as np
import datetime
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import slack

from .__init__ import pprint, rank


pathSave = '/local/scratch/altamura/analysis_results/bahamas_timing/'


def report_file(redshift: str) -> h5py.File:
	if rank==0:
		pathFile = '/local/scratch/altamura/analysis_results/alignment_project'
		if not os.path.exists(pathFile): os.makedirs(pathFile)
		h5file = h5py.File(os.path.join(pathFile, f"bahamas_hyd_alignment_{redshift}.hdf5"), 'w')
		return h5file

def error_file(redshift: str, errors: list) -> None:
	if rank==0:
		pathFile = '/local/scratch/altamura/analysis_results/alignment_project'
		if not os.path.exists(pathFile): os.makedirs(pathFile)
		with open(os.path.join(pathFile, f"bahamas_hyd_error_{redshift}.txt"), 'w') as e:
			for i in errors:
				print(f"{redshift}, {i}", file=e)


def fitFunc(t, a, b):
	return a*t+b

def redshift_str2num(z: str) -> float:
	"""
	Converts the redshift of the snapshot from text to numerical,
	in a format compatible with the file names.
	E.g. float z = 2.16 <--- str z = 'z002p160'.
	"""
	z = z.strip('z').replace('p', '.')
	return round(float(z), 3)

def time_checkpoint(start: datetime.datetime) -> float:
	end = datetime.datetime.now()
	elapsed = (end - start).total_seconds()
	return elapsed

def file_benchmarks(redshift: str) -> str:
	timing_filename = pathSave + f"bahamas_timing_{redshift}.txt"
	with open(timing_filename, "a") as benchmarks:
		pprint(f"#{redshift}", file=benchmarks)
	return timing_filename

def record_benchmarks(redshift: str, data: tuple):
	timing_filename = pathSave + f"bahamas_timing_{redshift}.txt"
	data = list(data)
	if data[0] == 'load': data[0] = 0
	elif data[0] == 'compute': data[0] = 1
	data = [f"{item}" for item in data]
	row = ','.join(data)
	# Print benckmarks to file
	with open(timing_filename, "a") as benchmarks:
		pprint(row, file=benchmarks)

def display_benchmarks(redshift: str):
	if rank == 0:
		timing_filename = pathSave+f"bahamas_timing_{redshift}.txt"
		plot_filename = pathSave+f"bahamas_timing_{redshift}.png"

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_ylim(0.1, 30)
		ax.set_xlabel('FOF cluster index')
		ax.set_ylabel('Computation time [seconds]')

		# Organise data and make halo_id start from 1 for log-scale plot
		lines = np.loadtxt(timing_filename, comments="#", delimiter=",", unpack=False).T
		tag = lines[0]
		lines[1] += 1
		n_load = lines[1][np.where(tag == 0)[0]]
		n_compute = lines[1][np.where(tag == 1)[0]]
		t_load = lines[2][np.where(tag==0)[0]]
		t_compute = lines[2][np.where(tag==1)[0]]
		n_tot = n_load
		t_tot = t_load+t_compute

		# Display raw data
		ax.scatter(n_load, t_load, marker='.', c='yellowgreen', s=3, alpha=0.3, label=f'z = {redshift_str2num(redshift)}, load')
		ax.scatter(n_compute, t_compute, marker='.', c='orchid', s=3, alpha=0.3, label=f'z = {redshift_str2num(redshift)}, compute')
		ax.scatter(n_tot, t_tot, marker='.', c='grey', s=3, alpha=0.3, label=f'z = {redshift_str2num(redshift)}, total')
		del n_load, t_load, n_compute, t_compute

		# Fit function to benchmarks
		n_fit = []
		t_fit = []
		for i in range(int(np.max(n_tot))):
			idx = np.where(n_tot == i)[0]
			if len(idx) == 1:
				n_fit.append(n_tot[idx][0])
				t_fit.append(t_tot[idx][0])
			elif len(idx) > 1:
				n_fit.append(np.mean(n_tot[idx]))
				t_fit.append(np.median(t_tot[idx]))

		# Make power-law fot
		n_fit = np.log10(np.asarray(n_fit))
		t_fit = np.log10(np.asarray(t_fit))
		fitParams, _ = curve_fit(fitFunc, n_fit, t_fit)
		n_display = np.logspace(0, np.log10(14400), 10)
		t_display = 10 ** fitFunc(np.log10(n_display), fitParams[0], fitParams[1])
		del n_fit, t_fit

		# Compute total computing time estimate
		eta_tot = np.sum(10**fitFunc(np.log10(np.linspace(1,14401,14401,dtype=np.int)), fitParams[0], fitParams[1]))
		eta_tot -= (eta_tot%60) # Round to minutes
		eta_tot = datetime.timedelta(seconds=eta_tot)
		ax.plot(n_display, t_display, color='red', label=f'z = {redshift_str2num(redshift)}, ETA = {eta_tot}')

		plt.legend()
		plt.savefig(plot_filename, dpi=300)

		# Send files to Slack: init slack client with access token
		print(f"[+] Forwarding {redshift} benchmarks to the `#personal` Slack channel...")
		slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
		client = slack.WebClient(token=slack_token)
		response = client.files_upload(
				file=plot_filename,
				initial_comment=f"This file was sent upon completion of the plot factory pipeline.\nAttachments: {plot_filename}",
				channels='#personal'
		)


