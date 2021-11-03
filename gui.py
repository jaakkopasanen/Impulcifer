import os
import tkinter
import re
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfilename
from tkinter.messagebox import showinfo
import recorder, impulcifer
import sounddevice

#tooltip for widgets
class ToolTip(object):
	def __init__(self, widget, text='widget info'):
		self.waittime = 500     #miliseconds
		self.wraplength = 180   #pixels
		self.widget = widget
		self.text = text
		self.widget.bind("<Enter>", self.enter)
		self.widget.bind("<Leave>", self.leave)
		self.widget.bind("<ButtonPress>", self.leave)
		self.id = None
		self.tw = None
	def enter(self, event=None):
		self.schedule()
	def leave(self, event=None):
		self.unschedule()
		self.hidetip()
	def schedule(self):
		self.unschedule()
		self.id = self.widget.after(self.waittime, self.showtip)
	def unschedule(self):
		id = self.id
		self.id = None
		if id:
			self.widget.after_cancel(id)
	def showtip(self, event=None):
		x = y = 0
		x, y, cx, cy = self.widget.bbox("insert")
		x += self.widget.winfo_rootx() + 25
		y += self.widget.winfo_rooty() + 20
		# creates a toplevel window
		self.tw = Toplevel(self.widget)
		# Leaves only the label and removes the app window
		self.tw.wm_overrideredirect(True)
		self.tw.wm_geometry("+%d+%d" % (x, y))
		label = Label(self.tw, text=self.text, justify='left',
					  background="#ffffff", relief='solid', borderwidth=1,
					  wraplength = self.wraplength)
		label.pack(ipadx=1)
	def hidetip(self):
		tw = self.tw
		self.tw= None
		if tw:
			tw.destroy()

#decimal entry validator
def validate_double(inp):
	if not inp or inp == '-':
		return True
	try:
		float(inp)
	except:
		return False
	return True

#integer entry validator
def validate_int(inp):
	if not inp:
		return True
	try:
		int(inp)
	except:
		return False
	if len(inp) > 5: #limit chars to 5
		return False
	if '-' in inp:
		return False
	return True

#open dir dialog
def opendir(var):
	path = askdirectory(initialdir=os.path.dirname(var.get()))
	if not path:
		return
	path = os.path.abspath(path) #make all separators the correct one
	path = path.replace(os.getcwd() + os.path.sep, '') #prefer relative paths when possible
	var.set(path)


#open file dialog
def openfile(var, filetypes):
	path = askopenfilename(initialdir=os.path.dirname(var.get()), initialfile=os.path.basename(var.get()), filetypes=filetypes)
	if not path:
		return
	path = os.path.abspath(path)
	path = path.replace(os.getcwd() + os.path.sep, '')
	var.set(path)

#save file dialog
def savefile(var):
	path = asksaveasfilename(initialdir=os.path.dirname(var.get()), initialfile=os.path.basename(var.get()), defaultextension=".wav", filetypes=(('WAV file', '*.wav'), ('All files', '*.*')))
	if not path:
		return
	path = os.path.abspath(path)
	path = path.replace(os.getcwd() + os.path.sep, '')
	var.set(path)

#pack widget into canvas
def pack(widget, samerow=False):
	if not samerow:
		pos[1] += widget.winfo_reqheight() + 5
		pos[0] = 10
	widget.place(x=pos[0], y=pos[1], anchor=W)
	widgetpos = (pos[0], pos[1])
	pos[0] += widget.winfo_reqwidth()
	global maxwidth
	maxwidth = max(maxwidth, pos[0])
	global maxheight
	maxheight = pos[1] + 20
	root.update()
	return widgetpos

#RECORDER WINDOW
root = Tk()

root.title('Recorder')
root.resizable(False, False)
canvas1 = Canvas(root)

pos = [0, 0]
maxwidth = 0
maxheight = 0

#refresh record window
def refresh1(init=False):
	host_apis = {}
	i = 0
	for host in sounddevice.query_hostapis():
		host_apis[i] = host['name']
		i += 1

	host_api_optionmenu['menu'].delete(0, 'end')
	for host in host_apis.values():
		host_api_optionmenu['menu'].add_command(label=host, command=tkinter._setit(host_api, host))

	if not host_apis:
		host_api.set('')
	elif init and 'Windows DirectSound' in host_apis.values():
		host_api.set('Windows DirectSound')
	elif host_api.get() not in host_apis.values():
		host_api.set(host_apis[0])

	output_devices = []
	input_devices = []
	for device in sounddevice.query_devices():
		if host_apis[device['hostapi']] == host_api.get():
			if device['max_output_channels'] > 0:
				output_devices.append(device['name'])
			elif device['max_input_channels'] > 0:
				input_devices.append(device['name'])
	output_device_optionmenu['menu'].delete(0, 'end')
	input_device_optionmenu['menu'].delete(0, 'end')
	for device in output_devices:
		output_device_optionmenu['menu'].add_command(label=device, command=tkinter._setit(output_device, device))
	for device in input_devices:
		input_device_optionmenu['menu'].add_command(label=device, command=tkinter._setit(input_device, device))
	if not output_devices:
		output_device.set('')
	elif output_device.get() not in output_devices:
		output_device.set(output_devices[0])
	if not input_devices:
		input_device.set('')
	elif input_device.get() not in input_devices:
		input_device.set(input_devices[0])

	channels_entry.config(state=NORMAL if channels_check.get() else DISABLED)

#playback device
output_device = StringVar()
output_device.trace('w', lambda *args: refresh1())
pack(Label(canvas1, text='Playback device'))
output_device_optionmenu = OptionMenu(canvas1, variable=output_device, value=None, command=refresh1)
pack(output_device_optionmenu, samerow=True)

#record device
input_device = StringVar()
input_device.trace('w', lambda *args: refresh1())
pack(Label(canvas1, text='Recording device'))
input_device_optionmenu = OptionMenu(canvas1, variable=input_device, value=None, command=refresh1)
pack(input_device_optionmenu, samerow=True)

#host API
pack(Label(canvas1, text='Host API'))
host_api = StringVar()
host_api.trace('w', lambda *args: refresh1())
host_api_optionmenu = OptionMenu(canvas1, host_api, value=None, command=refresh1)
pack(host_api_optionmenu, samerow=True)

#sound file to play
pack(Label(canvas1, text='File to play'))
play = StringVar(value=os.path.join('data', 'sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav'))
play_entry = Entry(canvas1, textvariable=play, width=70)
pack(play_entry)
pack(Button(canvas1, text='...', command=lambda: openfile(play, (('Audio files', '*.wav'), ('All files', '*.*')))), samerow=True)

#output file
pack(Label(canvas1, text='Record to file'))
record = StringVar(value=os.path.join('data', 'my_hrir', 'FL,FR.wav'))
record_entry = Entry(canvas1, textvariable=record, width=70)
pack(record_entry)
pack(Button(canvas1, text='...', command=lambda: savefile(record)), samerow=True)

#force number of channels
channels_check = BooleanVar()
channels_checkbutton = Checkbutton(canvas1, text="Force # of channels", variable=channels_check, command=refresh1)
pack(channels_checkbutton)
ToolTip(channels_checkbutton, 'For room correction: some measurement microphones like MiniDSP UMIK-1 are seen as stereo microphones by Windows and will for that reason record a stereo file. recorder.py can force the capture to be one channel')
channels = IntVar(value=1)
channels_entry = Entry(canvas1, textvariable=channels, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
pack(channels_entry, samerow=True)

#append
append = BooleanVar()
append_check = Checkbutton(canvas1, text="Append", variable=append)
ToolTip(append_check, 'Add track(s) to existing file. Silence will be added to end of each track to make all equal in length.')
pack(append_check)

#record button
def recordaction():
	recorder.play_and_record(play=play_entry.get(), record=record_entry.get(), input_device=input_device.get(), output_device=output_device.get(), host_api=host_api.get(), channels=(channels.get() if channels_check.get() else 2), append=append.get())
	showinfo('', 'Recorded to ' + record_entry.get())
pack(Button(canvas1, text='RECORD', command=recordaction))

refresh1(init=True)
root.geometry(str(maxwidth) + 'x' + str(maxheight) + '+0+0')
canvas1.config(width=maxwidth, height=maxheight)
canvas1.pack()

#IMPULCIFER WINDOW
maxwidth2 = maxwidth
maxheight2 = maxheight
maxwidth = 0
maxheight = 0
pos.clear()
pos += [0,0]
window2 = Toplevel(root)
window2.title('Impulcifer')
canvas2 = Canvas(window2)

#refresh impulcifer window
def refresh2(changedpath=False):
	if changedpath:
		if os.path.exists(dir_path.get()):
			files = os.listdir(dir_path.get().strip())
			if len(files) > 100: #don't want to scan a megafolder
				return
			s = ';'.join(files)
			if re.search(r"\broom(-[A-Z]{2}(,[A-Z]{2})*-(left|right))?\.wav\b", s, re.I):
				do_room_correction_msg.set('found room wav')
				do_room_correction_msg_label.config(foreground='green')
				do_room_correction.set(True)
			else:
				do_room_correction_msg.set('room wav not found!')
				do_room_correction_msg_label.config(foreground='red')
				# do_room_correction.set(False)
			if re.search(r'\bheadphones\.wav\b', s):
				do_headphone_compensation_msg.set('found headphones wav')
				do_headphone_compensation_msg_label.config(foreground='green')
				do_headphone_compensation.set(True)
			else:
				do_headphone_compensation_msg.set('headphones wav not found!')
				do_headphone_compensation_msg_label.config(foreground='red')
				# do_headphone_compensation.set(False)
			if re.search(r"\beq(-left|-right)?\.csv\b", s, re.I):
				do_equalization_msg.set('found eq csv')
				do_equalization_msg_label.config(foreground='green')
				do_equalization.set(True)
			else:
				do_equalization_msg.set('eq csv not found!')
				do_equalization_msg_label.config(foreground='red')
				# do_equalization.set(False)

	if do_room_correction.get():
		do_room_correction_msg_label.place(x=label_pos[do_room_correction_msg_label][0], y=label_pos[do_room_correction_msg_label][1], anchor=W)
	else:
		do_room_correction_msg_label.place_forget()
	if do_headphone_compensation.get():
		do_headphone_compensation_msg_label.place(x=label_pos[do_headphone_compensation_msg_label][0], y=label_pos[do_headphone_compensation_msg_label][1], anchor=W)
	else:
		do_headphone_compensation_msg_label.place_forget()
	if do_equalization.get():
		do_equalization_msg_label.place(x=label_pos[do_equalization_msg_label][0], y=label_pos[do_equalization_msg_label][1], anchor=W)
	else:
		do_equalization_msg_label.place_forget()

	specific_limit_entry.config(state=NORMAL if do_room_correction.get() else DISABLED)
	generic_limit_entry.config(state=NORMAL if do_room_correction.get() else DISABLED)
	room_target_entry.config(state=NORMAL if do_room_correction.get() else DISABLED)
	room_mic_calibration_entry.config(state=NORMAL if do_room_correction.get() else DISABLED)
	fr_combination_method_optionmenu.config(state=NORMAL if do_room_correction.get() else DISABLED)
	fs_optionmenu.config(state=NORMAL if fs_check.get() else DISABLED)
	decay_entry.config(state=DISABLED if decay_per_channel.get() else NORMAL)
	# if decay_per_channel.get():
	# 	decay.set('')

	if show_adv.get():
		for widget in adv_options_pos:
			widget.place(x=adv_options_pos[widget][0], y=adv_options_pos[widget][1], anchor=W)

		if channel_balance.get() == 'number':
			channel_balance_db_entry.place(x=adv_options_pos[channel_balance_db_entry][0], y=adv_options_pos[channel_balance_db_entry][1], anchor=W)
			channel_balance_db_label.place(x=adv_options_pos[channel_balance_db_label][0], y=adv_options_pos[channel_balance_db_label][1], anchor=W)
		else:
			channel_balance_db_entry.place_forget()
			channel_balance_db_label.place_forget()

		if decay_per_channel.get():
			for i in range(7):
				decay_labels[i].place(x=adv_options_pos[decay_labels[i]][0], y=adv_options_pos[decay_labels[i]][1], anchor=W)
				decay_entries[i].place(x=adv_options_pos[decay_entries[i]][0], y=adv_options_pos[decay_entries[i]][1], anchor=W)
		else:
			for i in range(7):
				decay_labels[i].place_forget()
				decay_entries[i].place_forget()
	else:
		for widget in adv_options_pos:
			widget.place_forget()

#your recordings
pack(Label(canvas2, text='Your recordings'))
dir_path = StringVar(value=os.path.join('data', 'my_hrir'))
dir_path.trace('w', lambda *args: refresh2(changedpath=True))
dir_path_entry = Entry(canvas2, textvariable=dir_path, width=80)
pack(dir_path_entry)
pack(Button(canvas2, text='...', command=lambda: opendir(dir_path)), samerow=True)

#test signal used
test_signal_label = Label(canvas2, text='Test signal used')
ToolTip(test_signal_label, 'Signal used in the measurement.')
pack(test_signal_label)
test_signal = StringVar(value=os.path.join('data', 'sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav'))
test_signal_entry = Entry(canvas2, textvariable=test_signal, width=80)
pack(test_signal_entry)
pack(Button(canvas2, text='...', command=lambda: openfile(test_signal, (('Audio files', '*.wav *.pkl'), ('All files', '*.*')))), samerow=True)

#room correction
label_pos = {}
do_room_correction = BooleanVar()
do_room_correction_checkbutton = Checkbutton(canvas2, text="Room correction ", variable=do_room_correction, command=refresh2)
ToolTip(do_room_correction_checkbutton, "Do room correction from room measurements in format room-<SPEAKERS>-<left|right>.wav located in your folder; e.g. room-FL,FR-left.wav. Generic measurements are named room.wav")
pack(do_room_correction_checkbutton)
do_room_correction_msg = StringVar()
do_room_correction_msg_label = Label(canvas2, textvariable=do_room_correction_msg)
label_pos[do_room_correction_msg_label] = pack(do_room_correction_msg_label, samerow=True)
specific_limit = IntVar(value=20000)
specific_limit_label = Label(canvas2, text='Specific Limit (Hz)')
ToolTip(specific_limit_label, "Upper limit for room equalization with speaker-ear specific room measurements. Equalization will drop down to 0 dB at this frequency in the leading octave.")
pack(specific_limit_label)
specific_limit_entry = Entry(canvas2, textvariable=specific_limit, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
pack(specific_limit_entry, samerow=True)
generic_limit = IntVar(value=1000)
genericlimitlabel = Label(canvas2, text='Generic Limit (Hz)')
ToolTip(genericlimitlabel, "Upper limit for room equalization with generic room measurements. Equalization will drop down to 0 dB at this frequency in the leading octave.")
pack(genericlimitlabel, samerow=True)
generic_limit_entry = Entry(canvas2, textvariable=generic_limit, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
pack(generic_limit_entry, samerow=True)
fr_combination_method_label = Label(canvas2, text='FR combination method')
pack(fr_combination_method_label, samerow=True)
fr_combination_methods = ['average', 'conservative']
fr_combination_method = StringVar(value=fr_combination_methods[0])
fr_combination_method_optionmenu = OptionMenu(canvas2, fr_combination_method, *fr_combination_methods)
ToolTip(fr_combination_method_label, 'Method for combining frequency responses of generic room measurements if there are more than one tracks in the file. "average" will simply average the frequency responses. "conservative" will take the minimum absolute value for each frequency but only if the values in all the measurements are positive or negative at the same time.')
pack(fr_combination_method_optionmenu, samerow=True)
room_mic_calibration_label = Label(canvas2, text='Mic calibration')
pack(room_mic_calibration_label)
room_mic_calibration = StringVar()
room_mic_calibration_entry = Entry(canvas2, textvariable=room_mic_calibration, width=65)
ToolTip(room_mic_calibration_label, 'Calibration data is subtracted from the room frequency responses. Uses room-mic-calibration.txt (or csv) by default if it exists.')
pack(room_mic_calibration_entry, samerow=True)
pack(Button(canvas2, text='...', command=lambda: openfile(room_mic_calibration, (('Text files', '*.csv *.txt'), ('All files', '*.*')))), samerow=True)
room_target_label = Label(canvas2, text='Target Curve')
pack(room_target_label)
room_target = StringVar()
room_target_entry = Entry(canvas2, textvariable=room_target, width=65)
ToolTip(room_target_label, 'Head related impulse responses will be equalized with the difference between room response measurements and room response target. Uses room-target.txt (or csv) by default if it exists.')
pack(room_target_entry, samerow=True)
pack(Button(canvas2, text='...', command=lambda: openfile(room_target, (('Text files', '*.csv *.txt'), ('All files', '*.*')))), samerow=True)

#headphone compensation
do_headphone_compensation = BooleanVar()
do_headphone_compensation_checkbutton = Checkbutton(canvas2, text="Headphone compensation ", variable=do_headphone_compensation, command=refresh2)
ToolTip(do_headphone_compensation_checkbutton, 'Equalize HRIR tracks with headphone compensation measurement headphones.wav')
pack(do_headphone_compensation_checkbutton)
do_headphone_compensation_msg = StringVar()
do_headphone_compensation_msg_label = Label(canvas2, textvariable=do_headphone_compensation_msg)
label_pos[do_headphone_compensation_msg_label] = pack(do_headphone_compensation_msg_label, samerow=True)

#headphone EQ
do_equalization = BooleanVar()
do_equalization_checkbutton = Checkbutton(canvas2, text="Custom EQ", variable=do_equalization, command=refresh2)
ToolTip(do_equalization_checkbutton, 'Read equalization FIR filter or CSV settings from file called eq.csv in your folder. The eq file must be an AutoEQ produced result CSV file. Separate equalizations are supported with files eq-left.csv and eq-right.csv.')
pack(do_equalization_checkbutton)
do_equalization_msg = StringVar()
do_equalization_msg_label = Label(canvas2, textvariable=do_equalization_msg)
label_pos[do_equalization_msg_label] = pack(do_equalization_msg_label, samerow=True)

#plot
plot = BooleanVar()
plot_checkbutton = Checkbutton(canvas2, text="Plot results", variable=plot, command=refresh2)
ToolTip(plot_checkbutton, 'Create graphs in your recordings folder (will increase processing time)')
pack(plot_checkbutton)

show_adv = BooleanVar()
pack(Checkbutton(canvas2, text='Advanced options', variable=show_adv, command=refresh2))
adv_options_pos = {} #save advanced options widgets' positions to show/hide

#resample
fs_check = BooleanVar()
fs_checkbutton = Checkbutton(canvas2, text="Resample to (Hz)", variable=fs_check, command=refresh2)
adv_options_pos[fs_checkbutton] = pack(fs_checkbutton)
sample_rates = [44100, 48000, 88200, 96000, 176400, 192000, 352000, 384000]
fs = IntVar(value=48000)
fs_optionmenu = OptionMenu(canvas2, fs, *sample_rates)
adv_options_pos[fs_optionmenu] = pack(fs_optionmenu, samerow=True)

#target level
target_level_label = Label(canvas2, text='Target level (dB)')
adv_options_pos[target_level_label] = pack(target_level_label)
target_level = StringVar()
target_level_entry = Entry(canvas2, textvariable=target_level, width=7, validate='key', vcmd=(root.register(validate_double), '%P'))
ToolTip(target_level_label, 'Normalize the average output BRIR level to the given numeric value. This makes it possible to compare HRIRs with somewhat similar loudness levels. Typically the desired level is several dB negative such as -12.5')
adv_options_pos[target_level_entry] = pack(target_level_entry, samerow=True)

#bass boost
bass_boost_gain_label = Label(canvas2, text='Bass boost (dB)')
ToolTip(bass_boost_gain_label, 'Bass boost shelf')
adv_options_pos[bass_boost_gain_label] = pack(bass_boost_gain_label)
bass_boost_gain = DoubleVar()
bass_boost_gain_entry = Entry(canvas2, textvariable=bass_boost_gain, width=7, validate='key', vcmd=(root.register(validate_double), '%P'))
ToolTip(bass_boost_gain_entry, 'Gain')
adv_options_pos[bass_boost_gain_entry] = pack(bass_boost_gain_entry, samerow=True)

bass_boost_fc_label = Label(canvas2, text='Fc')
adv_options_pos[bass_boost_fc_label] = pack(bass_boost_fc_label, samerow=True)
bass_boost_fc = IntVar(value=105)
bass_boost_fc_entry = Entry(canvas2, textvariable=bass_boost_fc, width=7, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[bass_boost_fc_entry] = pack(bass_boost_fc_entry, samerow=True)
ToolTip(bass_boost_fc_entry, 'Center Freq')

bass_boost_q_label = Label(canvas2, text='Q')
adv_options_pos[bass_boost_q_label] = pack(bass_boost_q_label, samerow=True)
bass_boost_q = DoubleVar(value=0.76)
bass_boost_q_entry = Entry(canvas2, textvariable=bass_boost_q, width=7, validate='key', vcmd=(root.register(validate_double), '%P'))
adv_options_pos[bass_boost_q_entry] = pack(bass_boost_q_entry, samerow=True)
ToolTip(bass_boost_q_entry, 'Quality')

#tilt
tilt_label = Label(canvas2, text='Tilt (dB)')
adv_options_pos[tilt_label] = pack(tilt_label)
tilt = DoubleVar()
tilt_entry = Entry(canvas2, textvariable=tilt, width=7, validate='key', vcmd=(root.register(validate_double), '%P'))
ToolTip(tilt_label, 'Target tilt in dB/octave. Positive value (upwards slope) will result in brighter frequency response and negative value (downwards slope) will result in darker frequency response.')
adv_options_pos[tilt_entry] = pack(tilt_entry, samerow=True)

#Channel Balance
channel_balance_label = Label(canvas2, text='Channel Balance')
adv_options_pos[channel_balance_label] = pack(channel_balance_label)
channel_balances = ['none', 'trend', 'mids', 'avg', 'min', 'left', 'right', 'number']
channel_balance = StringVar(value=channel_balances[0])
channel_balance.trace('w', lambda *args: refresh2())
channel_balance_optionmenu = OptionMenu(canvas2, channel_balance, *channel_balances)
adv_options_pos[channel_balance_optionmenu] = pack(channel_balance_optionmenu, samerow=True)
ToolTip(channel_balance_label, 'Channel balance correction by equalizing left and right ear results to the same level or frequency response. "trend" equalizes right side by the difference trend of right and left side. "left" equalizes right side to left side fr, "right" equalizes left side to right side fr, "avg" equalizes both to the average fr, "min" equalizes both to the minimum of left and right side frs. Number values will boost or attenuate right side relative to left side by the number of dBs. "mids" is the same as the numerical values but guesses the value automatically from mid frequency levels.')
channel_balance_db = IntVar(value=0)
channel_balance_db_entry = Entry(canvas2, textvariable=channel_balance_db, width=5, validate='key', vcmd=(root.register(validate_double), '%P'))
adv_options_pos[channel_balance_db_entry] = pack(channel_balance_db_entry, samerow=True)
channel_balance_db_label = Label(canvas2, text='dB')
adv_options_pos[channel_balance_db_label] = pack(channel_balance_db_label, samerow=True)

#decay
decay_label = Label(canvas2, text='Decay (ms)')
adv_options_pos[decay_label] = pack(decay_label)
decay = StringVar()
decay_entry = Entry(canvas2, textvariable=decay, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
ToolTip(decay_label, 'Target decay time to reach -60 dB. When natural decay time is longer than the target decay time, a downward slope will be applied to decay tail. Decay cannot be increased with this. Can help reduce ringing in the room without having to do any physical room treatments.')
adv_options_pos[decay_entry] = pack(decay_entry, samerow=True)
decay_per_channel = BooleanVar()
decay_per_channel_checkbutton = Checkbutton(canvas2, text="per channel", variable=decay_per_channel, command=refresh2)
adv_options_pos[decay_per_channel_checkbutton] = pack(decay_per_channel_checkbutton, samerow=True)

decay_fl_label = Label(canvas2, text='FL')
adv_options_pos[decay_fl_label] = pack(decay_fl_label, samerow=True)
decay_fl = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_fl] = pack(decay_fl, samerow=True)
decay_fc_label = Label(canvas2, text='FC')
adv_options_pos[decay_fc_label] = pack(decay_fc_label, samerow=True)
decay_fc = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_fc] = pack(decay_fc, samerow=True)
decay_fr_label = Label(canvas2, text='FR')
adv_options_pos[decay_fr_label] = pack(decay_fr_label, samerow=True)
decay_fr = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_fr] = pack(decay_fr, samerow=True)
decay_sl_label = Label(canvas2, text='SL')
adv_options_pos[decay_sl_label] = pack(decay_sl_label, samerow=True)
decay_sl = Entry(canvas2,  width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_sl] = pack(decay_sl, samerow=True)
decay_sr_label = Label(canvas2, text='SR')
adv_options_pos[decay_sr_label] = pack(decay_sr_label, samerow=True)
decay_sr = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_sr] = pack(decay_sr, samerow=True)
decay_bl_label = Label(canvas2, text='BL')
adv_options_pos[decay_bl_label] = pack(decay_bl_label, samerow=True)
decay_bl = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_bl] = pack(decay_bl, samerow=True)
decay_br_label = Label(canvas2, text='BR')
adv_options_pos[decay_br_label] = pack(decay_br_label, samerow=True)
decay_br = Entry(canvas2, width=5, validate='key', vcmd=(root.register(validate_int), '%P'))
adv_options_pos[decay_br] = pack(decay_br, samerow=True)

decay_labels = []
decay_entries = []
decay_labels.append(decay_fl_label)
decay_labels.append(decay_fc_label)
decay_labels.append(decay_fr_label)
decay_labels.append(decay_sl_label)
decay_labels.append(decay_sr_label)
decay_labels.append(decay_bl_label)
decay_labels.append(decay_br_label)
decay_entries.append(decay_fl)
decay_entries.append(decay_fc)
decay_entries.append(decay_fr)
decay_entries.append(decay_sl)
decay_entries.append(decay_sr)
decay_entries.append(decay_bl)
decay_entries.append(decay_br)

#impulcify button
def impulcifyaction():
	args = {'dir_path': dir_path.get(), 'test_signal': test_signal.get(), 'plot':plot.get(), 'do_room_correction': do_room_correction.get(), 'do_headphone_compensation':do_headphone_compensation.get(), 'do_equalization':do_equalization.get()}
	if do_headphone_compensation.get():
		args['room_target'] = room_target.get() if room_target.get() else None
		args['room_mic_calibration'] = room_mic_calibration.get() if room_mic_calibration.get() else None
		args['specific_limit'] = specific_limit.get()
		args['generic_limit'] = generic_limit.get()
		args['fr_combination_method'] = fr_combination_method.get()
	if show_adv.get():
		args['fs'] = fs.get() if fs_check.get() else None
		args['target_level'] = float(target_level.get()) if target_level.get() else None
		args['channel_balance'] = channel_balance_db.get() if channel_balance.get() == 'number' else (channel_balance.get() if channel_balance.get() != 'none' else None)
		args['bass_boost_gain'] = bass_boost_gain.get()
		args['bass_boost_fc'] = bass_boost_fc.get()
		args['bass_boost_q'] = bass_boost_q.get()
		args['tilt'] = tilt.get()
		if decay_per_channel.get():
			args['decay'] = {decay_labels[i].cget('text') : float(decay_entries[i].get()) / 1000 for i in range(7) if decay_entries[i].get()}
		elif decay.get():
			args['decay'] = {decay_labels[i].cget('text') : float(decay.get()) / 1000 for i in range(7)}
	print(args) #debug args
	impulcifer.main(**args)
	showinfo('Done!', 'Generated files, check recordings folder.')
pack(Button(canvas2, text='GENERATE', command=impulcifyaction))

canvas2.config(width=maxwidth, height=maxheight)
canvas2.pack()
window2.geometry(str(maxwidth) + 'x' + str(maxheight) + '+' + str(maxwidth2) + '+0')
window2.resizable(False, False)
refresh2(changedpath=True)
root.mainloop()