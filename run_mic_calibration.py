#
#  "run_mic_calibration.py" executable script runs "mic_calibration.py" from Impulcifer in a semi-guided wizard format.
#    source:  https://github.com/jaakkopasanen/Impulcifer/tree/master/research/mic-calibration
#
#                               HOW TO USE:
#
#  NOTE: this script will ask for command line "inputs" such as pressing Enter or typing a letter and then hitting Enter
#
#   1- You must have Impulcifer installed in some Python 3.8 or 3.9 environment:
#          all instructions here:   https://github.com/jaakkopasanen/Impulcifer
#   2- First "open/run" this "run_mic_calibration.py" file in Python environment where Impulcifer is installed.
#       You will see printouts of all AUDIO DEVICES that will be useful next (then just let the script error-out).
#   3- Modify the Audio Device settings in the "Settings to modify:" section (use info from AUDIO DEVICE printouts).
#   4- "open/run" this "run_mic_calibration.py" file in Python environment where Impulcifer is installed.
#       The script should guide you and prompt to either press Enter or inout a letter to continue.
#       if you get an error - read the last printouts before the error!
#   5- You can always skip some or all measurements OR go straight to calibration calculation.
#
#   tip - do measurements in 3+ locations with perhaps 2-4 rotations around Reference-microphone axis in each location.
#
#    license - the same as Impulcifer - MIT License.
#       v0.99    2022-02-22     kind-of ready (by Sergejs Dombrovskis)
#       v1.00    2022-02-23     actually working First version. Tested on Windows 10. (by Sergejs Dombrovskis)
#       v1.01    2022-02-24     added sound device printouts to simplify configuration (by S.D.)
#

#   - known issue - script gets stuck in Spyder debugger (but running normally in other modes and other consoles)

"""    Settings to modify:   """

# ###  modify the Device list as needed (but __FIRST__ run the script to see the list of available devices)
#           (see more info down below - search script for "List all audio Devices" ):

ref_input_device = "Microphone (Umik MME"  # remember to include "Audio  API"
targ_input_device = "Microphone (H2n MME"  # remember to include "Audio  API"
output_device = "DENON-AVR (NVIDIA MME"  # remember to include "Audio  API"


# leave on Autodetect OR manually set path to your Impulcifer folder (where "recorder.py" is):
Impulcifer_Folder = 'AutoDetect Path!'  # 'AutoDetect Path!' works if this script is right next to "impulcifer.py")
# Impulcifer_Folder = r"C:\Git_Impulcifier\Impulcifer"      # hard coded path input here

# ##################################################################################  end of Settings.


"""     List all audio Devices      """
# NOTE: when you later specify sound device, it will be used for SEARCHING, so it is enough to specify in one string:
#     • the beginning of the device name and 
#     • add some letters in the end to identify “host_api”

# For example, if “python -m sounddevice” gives you output:
"""	>  4 Microphone (Umik-1  Gain: 18dB , MME (2 in, 0 out)             """
# then you can specify this input_device as:
"""   "Microphone (Umi MME"                                             """
# where “ MME” is the host_api search string which could alo be “ WASAPI”, “ DirectSound” or other valid API.


# # #  following sound device code is based on Impulcifer "gui.py" by https://github.com/godlovesdavid
import sounddevice
# noinspection PyUnreachableCode
if False:   # this is the true full list of audio devices, with details
    sounddevice.query_devices()

output_devices = []
input_devices = []
Skip_These = ['Microsoft Sound Mapper - Output', 'Primary Sound Driver']  # junk options
host_apis = {}
i = 0

print('\n----- Audio  API  list -----          (for reference):')
for host in sounddevice.query_hostapis():
    print(host['name'])
    host_apis[i] = host['name']
    i += 1
print('\n----- Audio Output devices list ----- (for reference):')
for device in sounddevice.query_devices():
    if host_apis[device['hostapi']] != 'Windows WDM-KS' and device['name'] not in Skip_These:  # skipping junk
        if device['max_output_channels'] > 0 and not device['name'].startswith(tuple(output_devices)):
            output_devices.append(device['name'])
            print(device['name'])
print('\n----- Audio  Input devices list ----- (for reference):')
for device in sounddevice.query_devices():
    if host_apis[device['hostapi']] != 'Windows WDM-KS' and device['name'] not in Skip_These:  # skipping junk
        if device['max_input_channels'] > 0 and not device['name'].startswith(tuple(input_devices)):
            input_devices.append(device['name'])
            print(device['name'])
del input_devices, output_devices, device, Skip_These, host, i, host_apis   # cleaning up




"""         Import & init everything        """
import sys
import os
# from tkinter import *       #  to use message boxes at least Python 3.9 is required...

if Impulcifer_Folder == 'AutoDetect Path!':
    Impulcifer_Folder = os.path.abspath(os.path.join(__file__, os.pardir))  # Autodetect path (assuming correct place)

# note down folder of the "mic_calibration.py" file
Result_Folder = os.path.join(Impulcifer_Folder, 'research', 'mic-calibration')
# RELATIVE Path with sound sweep for the LEFT speaker
file_toPlay = os.path.join(Impulcifer_Folder, "data", "sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav")
# file_asRef = os.path.join(Impulcifer_Folder, "data", "sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav")
file_asRef = os.path.join(Impulcifer_Folder, "data", "sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl")

# change to Impulcifer folder (where "recorder.py" is):
if os.path.isdir(Impulcifer_Folder):
    # # os.chdir(Impulcifer_Folder)     os.getcwd()
    sys.path.insert(1, Impulcifer_Folder)   # add main   "Impulcifer"   folder to search path
    sys.path.insert(1, Result_Folder)       # add the "mic-calibration" folder to search path

else:
    print('\n\n\n\n_______STOP! please modify settings in this file to continue:_______\n')
    print(' Please change "Impulcifer_Folder" to the correct path to your Impulcifer folder')
    input("  -press Enter- to exit mic calibration (read reason above).")
    raise Exception('Please restart the script when you updated "Impulcifer_Folder" path in this script')

import sounddevice
import recorder                # Load   "recorder.py"   from Impulcifier
import mic_calibration  # Load   "mic_calibration.py"   from Impulcifier
# noinspection PyUnresolvedReferences
from impulse_response_estimator import ImpulseResponseEstimator   # C R I T I C A L  -  Do NOT Remove this LINE !!!





"""     Actual code, - just run the script!     """

#  configuration info printouts
print('\n   Measurements will use LEFT speaker connected to')
print('* loudspeaker output device: "' + output_device + '"')
print('* Reference microphone is:   "' + ref_input_device + '"')
print('* Target    microphone is:   "' + targ_input_device + '"')
if os.path.isfile(os.path.join(Result_Folder, 'room-mic-calibration.txt')) or \
        os.path.isfile(os.path.join(Result_Folder, 'room-mic-calibration.csv')):
    print('__ok, Reference microphone calibration file is found. \n')
else:
    print('__NO calibration file for Reference microphone was found! \n')


# check that there are no previous results
if os.path.isfile(os.path.join(Result_Folder, 'Results.png')) or \
        os.path.isfile(os.path.join(Result_Folder, 'right-mic-calibration.csv')) or \
        os.path.isfile(os.path.join(Result_Folder, 'left-mic-calibration.csv')):
    print('\n\n\n\n_______STOP! calibration output files already EXIST in folder:_______\n')
    print('   "' + Result_Folder + '"')
    print('    (one or more of these: "Results.png", "right-mic-calibration.csv", "left-mic-calibration.csv")')
    input("  -press Enter- to exit mic calibration (read reason above).")
    raise Exception('Please restart the script when you removed the previous result files.')


# # # ask if we should skip sound devices and measurements
# # # dialogAns = messagebox.askyesno("Mic Calibration wizard", "Ready to start choose:\nYes= go-to Measurements\n
#       No = jump directly to CALIBRATION calculations")


user_in = input('?  press Enter to go-to Measurements  |  type "x" to jump directly to CALIBRATION calculations')
if user_in.startswith('x'):
    print('\nJumping over measurements...')
    DoMeasurements = False
else:
    DoMeasurements = True


# to error out early in case of problems with sound devices:
if DoMeasurements:
    # check sound devices
    print('hint - checking that sound devices are valid and uniquely defined:')
    print('hint - If you just connected a device - you might need to restart Python')
    print('hint - If device is not found - enter a valid search string (use command "sounddevice.query_devices()")')
    print('hint - If more than one device is found - be more specific in the search string')
    detailed_Info = sounddevice.query_devices(ref_input_device)
    # noinspection PyRedeclaration
    detailed_Info = sounddevice.query_devices(targ_input_device)
    # noinspection PyRedeclaration
    detailed_Info = sounddevice.query_devices(output_device)
    print('Audio device check OK. good to go (assuming levels are good)')


# measurements LOOP for all measurements until user STOPs it.
measurement_nr = 0  # init
while DoMeasurements:
    measurement_nr += 1  # count next measurement
    overwrite_ok = False    # init
       
    
    # prompts for REFERENCE measurements
    if os.path.isfile(os.path.join(Result_Folder, 'room-' + str(measurement_nr) + '.wav')):
        print('\n\n WARNING! file "room-' + str(measurement_nr) + '.wav" for REFERENCE mic sweep number ' +
              str(measurement_nr) + ' already EXISTS!')
        user_in = input('?  press Enter to RE-measure  |  type "x" to STOP  |  type "j" to Jump over this measurements')
        overwrite_ok = False    # init
    else:
        print('\n ----- ready for REFERENCE mic sweep number ' + str(measurement_nr), ' -----')
        user_in = input('?  press Enter to START  |  type "x" to STOP  |  type "j" to Jump over this measurements')
        overwrite_ok = True    # init
        
    if user_in.startswith('x'):
        print('\nStop command detected - finished taking measurements')
        break
    elif user_in.startswith('j'):
        print('\nJump over command detected - skipping this measurement')    
    else:
        if not overwrite_ok:
            print('\n\n  Are You Sure you want to OVERWRITE "room-' + str(measurement_nr) + '.wav" file?')
            user_in = input('?  type "o" to Overwrite  |  press Enter to reconsider')
            if user_in.startswith('o'):
                print('\n   Overwrite command detected - removing old file')
                try:
                    os.remove(os.path.join(Result_Folder, 'room-' + str(measurement_nr) + '.wav'))
                except OSError:
                    pass
            else:
                print('\n   Reconsidering...')
                measurement_nr -= 1  # return counter back      
                continue

        # do the recording
        print('\n   Recording (be quiet...)')
        recorder.play_and_record(
            play=file_toPlay,
            record=os.path.join(Result_Folder, 'room-' + str(measurement_nr) + '.wav'),
            input_device=targ_input_device,
            output_device=output_device,
            host_api=None,
            channels=1,     # this must be forced to 1 channel !!!!!!!!!!!! (costed me a few hours of my life.)
            append=False)
    
    
    
    
    # prompts for TARGET measurements
    if os.path.isfile(os.path.join(Result_Folder, 'binaural-' + str(measurement_nr) + '.wav')):
        print('\n\n WARNING! file "binaural-' + str(measurement_nr) + '.wav" for TARGET mic sweep number ' +
              str(measurement_nr) + ' already EXISTS!')
        user_in = input('?  press Enter to RE-measure  |  type "x" to STOP  |  type "j" to Jump over this measurements')
        overwrite_ok = False    # init
    else:
        print('\n ===== ready for TARGET mic sweep number ' + str(measurement_nr), ' =====')
        user_in = input('?  press Enter to START  |  type "x" to STOP  |  type "j" to Jump over this measurements')
        overwrite_ok = True    # init
        
    if user_in.startswith('x'):
        print('\nStop command detected - finished taking measurements')
        break
    elif user_in.startswith('j'):
        print('\nJump over command detected - skipping this measurement')    
    else:
        if not overwrite_ok:
            print('\n\n  Are You Sure you want to OVERWRITE "binaural-' + str(measurement_nr) + '.wav" file?')
            user_in = input('?  type "o" to Overwrite  |  press Enter to reconsider')
            if user_in.startswith('o'):
                print('\n   Overwrite command detected - removing old file')
                try:
                    os.remove(os.path.join(Result_Folder, 'binaural-' + str(measurement_nr) + '.wav'))
                except OSError:
                    pass
            else:
                print('\n   Reconsidering...')
                measurement_nr -= 1  # return counter back      
                continue

        # do the recording
        print('\n   Recording (be quiet...)')
        recorder.play_and_record(
            play=file_toPlay,
            record=os.path.join(Result_Folder, 'binaural-' + str(measurement_nr) + '.wav'),
            input_device=targ_input_device,
            output_device=output_device,
            host_api=None,
            channels=2,
            append=False)
    
       

    
print('\n\n ########### ready to run CALIBRATION calculations ###########')
user_in = input('?  press Enter to START  |  type "x" to exit without finishing calibration')
    
if user_in.startswith('x'):
    print('\nStop command detected - exit without finishing calibration')
    input("  -press Enter- to exit mic calibration (calculation aborted).")
 
else:
    # ###  Call calibration function:
    #  python mic_calibration.py --test_signal="../../data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl"    
    #  mic_calibration.main("../../data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl")
    print('  Note - this script will hang open until you close the mic_calibration plot."')
    mic_calibration.main(file_asRef)

    print('\n\n__________DONE, results were saved in folder:\n')
    print('  "' + Result_Folder + '"')
    print('  see files: "Results.png", "right-mic-calibration.csv" and "left-mic-calibration.csv"')
    input("  -press Enter- to exit mic calibration (everything is Finished).")
