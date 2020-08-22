# Impulcifer
Impulcifer is a tool for creating head related impulse responses (HRIR) for speaker virtualization on headphones.

Normally headphones sound inside your head which is a clear disadvantage for games and movies but also for music
because basically all material has been created for speakers. Virtual surround technologies for headphones have existed
for a some time by now but almost all of them fail to fulfill expectations of out of head sound localization and the
naturalness of speakers. This is because your brains have learned to localize sounds only with your ears and head and
not with anyone else's. Surround sound on headphones can only be convincing when the surround virtualization technology
has been tailored for your ears. HRIR is the tailored model for supreme virtual speaker surround on headphones. When
done right virtualized speakers can be indistinguishable from real speakers.

Watch these videos to get an idea what good HRIRs can do. The method used by Smyth Realizer and Creative Super X-Fi
demos is the same what Impulcifer uses.

- [Realiser A16 Smyth Research (Kickstarter project)](https://www.youtube.com/watch?v=3mZhN3OG-tc)
- [a16 realiser](https://www.youtube.com/watch?v=RtY9QIkRJxA)
- [Creative Super X-Fi 3D Immersive Headphone Technology at CES 2018](https://www.youtube.com/watch?v=mAidEm9_JYM)

These demos are trying to make headphones sound as much as possible like the speakers they have in the demo room for a
good [wow](https://www.youtube.com/watch?v=KlLMlJ2tDkg) effect. Impulcifer actually takes this even further because
Impulcifer can do measurements with only one speaker so you don't need access to surround speaker setup and can do room
acoustic corrections which are normally not possible in real rooms with DSP.

## Installing
Impulcifer is used from a command line and requires some prerequisites. These installation instructions will guide you
through installing everything that is needed to run Impulcifer on you own PC.

- Download and install Git: https://git-scm.com/downloads. When installing Git on Windows, use Windows SSL verification
instead of Open SSL or you might run into problems when installing project dependencies.
- Download and install 64-bit [Python 3](https://www.python.org/getit/). Python 3.8 doesn't work yet. Make sure to check
*Add Python 3 to PATH*.

Rest will be done in terminal / command prompt. On Windows you'll find it by searching `cmd` in Start menu.
You should be able to simply copy and paste in these commands. 

- Install virtualenv.  
```bash
pip install virtualenv
```
- Git clone Impulcifer. This will create a folder in your home folder called `Impulcifer`. See [Updating](#updating)
for other versions than the latest.  
```bash
git clone https://github.com/jaakkopasanen/Impulcifer
```
- Go to Impulcifer folder.  
```bash
cd Impulcifer
```
- Create virtual environment for the project.  
```bash
virtualenv venv
```
- Activate virtualenv.  
```bash
# On Windows
venv\Scripts\activate
# On Mac and Linux
source venv/bin/activate
```
- Install required packages.  
```bash
pip install -U -r requirements.txt
```
- Verify installation. You should see help printed if everything went well.  
```bash
python impulcifer.py --help
```

When coming back at a later time you'll only need to activate virtual environment again before using Impulcifer.
```bash
cd Impulcifer
# On Windows
venv\Scripts\activate
# On Mac and Linux
source venv/bin/activate
```

### Updating
Impulcifer is under active development and updates quite frequently. Take a look at the [Changelog](./CHANGELOG.md) to
see what has changed.

Versions in Changelog have Git tags with which you can switch to another version than the latest one:
```bash
# Check available versions
git tag
# Update to a specific version
git checkout 1.0.0
```

You can update your own copy to the latest versions by running:
```bash
git checkout master
git pull
```

required packages change quite rarely but sometimes they do and then it's necessary to upgrade them
```bash
pip install -U -r requirements.txt
```
You can always invoke the update for required packages, it does no harm when nothing has changed.

## Demo
The actual HRIR measurements require a little investment in measurement gear and the chances are that you're here before
you have acquired them. There is a demo available for testing out Impulcifer without having to do the actual
measurements. `data/demo` folder contains five measurement files which are needed for running Impulcifer.
`headphones.wav` has the sine sweep recordings done with headphones and all the rest files are recordings done with
stereo speakers in multiple stages.

You can try out what Impulcifer does by running:
```bash
python impulcifer.py --test_signal=data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl --dir_path=data/demo 
```
Impulcifer will now process the measurements and produce `hrir.wav` and `hesuvi.wav` which can be used with headphone
speaker virtualization software such as [HeSuVi](https://sourceforge.net/projects/hesuvi/) to make headphones sound like
speakers in a room. When testing with HeSuVi copy `hesuvi.wav` into `C:\Program Files\Equalizer APO\config\Hesuvi\hrir`,
(re)start HeSuVi and select `hesuvi.wav` from the Common HRIRs list on Virtualization tab.

## Measurement
HRIR measurements are done with binaural microphones which are also called ear canal blocking microphones or in-ear
microphones. Exponential sine sweep test signal is played on speakers and the sound is recorded with the microphones at
ear canal openings. This setup ensures that the sound measured by the microphones is affected by the room, your body,
head and ears just like it is when listening to music playing on speakers. Impulcifer will then transform these
recordings into impulse responses, one per each speaker-ear pair.

Guide for doing the measurements yourself and comments about the gear needed to do it can be found in
[measurements](https://github.com/jaakkopasanen/Impulcifer/wiki/Measurements) page of Impulcifer wiki. The whole process
is really quite simple and doesn't take more than couple of minutes. Reading through the measurement guide is most
strongly recommended when doing measurements the first time or using a different speaker configuration the first time.

Following is a quick reference for running the measurements once you're familiar with the process. If you always use
`my_hrir` as the temporary folder and rename it after the processing has been done, you don't have to change the
following commands at all and you can simply copy-paste them for super quick process.

### 7.1 Speaker Setup
Steps and commands for doing measurements with 7.1 surround system:
- Put microphones in ears, put headphones on and run
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/headphones.wav"
```
- Take heaphones off, look forward and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FC,FR,SR,BR,BL,SL-7.1-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FL,FC,FR,SR,BR,BL,SL.wav"
```
- Process recordings  
```bash
python impulcifer.py --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl" --dir_path="data/my_hrir"
```

### Stereo Speaker Setup
Steps and commands for doing measurements with two speakers in four stages:
- Put microphones in ears, put headphones on and run
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/headphones.wav"
```
- Take heaphones off, look forward and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FL,FR.wav"
```
- Look 120 degrees left (left speaker should be on your right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/SR,BR.wav"
```
- Look 120 degrees right (right speaker should be on your left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/BL,SL.wav"
```
- Look directly at the left (or right) speaker and run either one of these commands  
```bash
# Using left speaker 
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FC.wav"
# Using right speaker 
python recorder.py --play="data/sweep-seg-FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FC.wav"
```
- Process recordings  
```bash
python impulcifer.py --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl" --dir_path="data/my_hrir" 
```

### Single Speaker
Steps and command for doing measurements with just a single speaker in 7 steps. All speaker measurements use either
`sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` or
`sweep-seg-FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` depending if the speaker is connected to left or right
cable terminals in the amplifier. This commands assume the speaker is connected to left speaker terminals.
- Put microphones in ears, put headphones on and run
```bash
python recorder.py --play="data/sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/headphones.wav"
```
- Look 30 degrees right of the speaker (speaker on your front left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FL.wav"
```
- Look directly at the speaker and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FC.wav"
```
- Look 30 degrees left of the speaker (speaker on you front right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/FR.wav"
```
- Look 90 degrees left of the speaker (speaker on your right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/SR.wav"
```
- Look 150 degrees left of the speaker (speaker on your back right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/BR.wav"
```
- Look 150 degrees right of the speaker (speaker on you back left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/BL.wav"
```
- Look 90 degrees right of the speaker (speaker on your left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="data/my_hrir/SL.wav"
```
- Process recordings  
```bash
python impulcifer.py --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl" --dir_path="data/my_hrir"
```

## Processing
Once you have obtained the sine sweep recordings, you can turn them into a HRIR file with Impulcifer. All the processing
is done by running a single command on command line. The command below assumes you have made a speaker recordings and
a headphones recording and saved the recording files into `data/my_hrir` folder. Start command prompt, jump to
Impulcifer folder and activate the virtual environment as described in the installation instructions if you don't have
command prompt open yet. Sine sweep recordings are processed by running `impulcifer.py` with Python as shown below.
```bash
python impulcifer.py --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl" --dir_path="data/my_hrir" --plot
```

You should have several WAV files and graphs in the folder. `hesuvi.wav` can now be used with HeSuVi to make your
headphones sound like speakers.

`--dir_path=data/my_hrir` tells Impulcifer that the recordings can be found in a folder called `my_hrir` under `data`.
Impulcifer will also write all the output files into this folder.

Impulcifer always needs to know which sine sweep signal was used during recording process. Test signal can be either a
WAV (`.wav`) file or a Pickle (`.pkl`) file. Test signal is read from a file called `test.pkl` or `test.wav`. 
`impulse_response_estimator.py` produces both but using a Pickle file is a bit faster. Pickle file however cannot be
used with `recorder.py`. An alternative way of passing the test signal is with a command line argument `--test_signal`
which takes is a path to the file eg. `--test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.pkl"`.

Sine sweep recordings are read from WAV files which have speaker names separated with commas and `.wav` extension eg.
`FL,FR.wav`. The individual speakers in the given file must be recorded in the order of the speaker names in the file
name. There can be multiple files if the recording was done with multiple steps as is the case when recording 7.1 setup
with two speakers. In that case there should be `FL,FR.wav`, `SR,BR.wav`, `BL,SL.wav` and `FC.wav` files in the folder.

#### Room Correction
Similar pattern is used for room acoustics measurements. The idea is to measure room response with a calibrated
measurement microphone in the exact same spot where the binaural microphones were. Room measurement files have file name
format of `room-<SPEAKERS>-<left|right>.wav`, where `<SPEAKERS>` is the comma separated list of speaker names and
`<left|right>` is either "left" or "right". This tells if the measurement microphone is measuring at the left or right
ear position. File name could be for example `room-FL,FR-left.wav`. Impulcifer does not support stereo measurement
microphones because vast majority of measurement microphones are mono. Room recording files need to be single track
only. Some measurement microphones like MiniDSP UMIK-1 are seen as stereo microphones by Windows and will for that
reason record a stereo file. `recorder.py` can force the capture to be one channel by setting `--channels=1`.

Generic room measurements can be done for speakers with which it's hard to position the measurement microphone
correctly. Impulcifer reads these measurements from `room.wav` file which can contain any number of tracks and any
number of sweeps per track. All the sweeps are being read and their frequency responses are combined. The combined
frequency response is used for room correction with the speakers that don't have specific measurements
(`room-FL,FR-left.wav` etc...).

There are two methods for combining the frequency responses: `"average"` and `"conservative"`. Average method takes the
average frequency response of all the measurements and builds the room correction equalization with that. Conservative
method takes the absolute minimum error for each frequency and only if all the measurements are on the same side of 0
level at the given frequency. This ensures that there will never be room correction adjustments that would make the
frequency response of any of the measurements worse. These methods are available with `--fr_combination_method=average`
and `--fr_combination_method=conservative`.

Upper frequency limit for room measurements can be adjusted with parameters `--specific_limit` and `--generic_limit`.
These will limit the room correction equalization to 0 dB above that frequency. This can be useful for avoiding pitfalls
of room correction in high frequencies. In theory there should not be need to limit room correction with speaker-ear
specific measurements if the measurement microphone was placed *exactly* in the same location as where the binaural
microphone was but in practice this might not be the case always. Especially the frequencies above 10 kHz are very
tricky and the gain is affected by a lot of small factors.

Generic room measurements are not expected to be in the same location as the binaural microphones were so limiting the
equalization to less than 1 kHz is probably a good idea. Conservative combination method with several measurements is
safer method and with that it should be safer to try to increase the limit up from 1 kHz. For example
`--specific_limit=5000 --generic_limit=2000` would ensure that room correction won't adjust frequency response of HRIR
above 5 kHz for any speaker-ear pairs and above 2 kHz for speaker-ear pairs that don't have specific room measurements.

Room measurements can be calibrated with the measurement microphone calibration file called `room-mic-calibration.txt`
or `room-mic-calibration.csv`. This must be a CSV file where the first column contains frequencies and the second one
amplitude data. Data is expected to be calibration data and not equalization data. This means that the calibration data
is subtracted from the room frequency responses. An alternative way of passing in the measurement microphone calibration
file is with a command line argument `--room_mic_calibration` and it takes a path to the file eg.
`--room_mic_calibration="data/umik-1_90deg.txt"`

Room frequency response target is read from a file called `room-target.txt` or `room-target.csv`. Head related impulse
responses will be equalized with the difference between room response measurements and room response target. An
alternative way to pass in the target file is with a command line argument `--room_target` eg.
`--room_target="data/harman-in-room-loudspeaker-target.csv"`.

Room correction can be skipped by adding a command line argument `--no_room_correction` without any value.

#### Headphone Compensation
Impulcifer will compensate for the headphone frequency response using headphone sine sweep recording if the folder
contains file called `headphones.wav`. If you have the file but would like not to have headphone compensation, you can
simply rename the file for example as `headphones.wav.bak` and run the command again. 

Headphone equalization can be baked into the produced HRIR file by having a file called `eq.csv` in the folder. The eq 
file must be an AutoEQ produced result CSV file. Separate equalizations for left and right channels are supported with
files `eq-left.csv` and `eq-right.csv`. Headphone equalization is useful for in-ear monitors because it's not possible
to do headphone compensation with IEMs. When using IEMS you still need an around ear headphone for the headphone
compensation. **eq.wav is no longer supported!**

Impulcifer will bake the frequency response transformation from the CSV file into the HRIR and you can enjoy speaker
sound with your IEMs. You can generate this filter with [AutoEQ](https://github.com/jaakkopasanen/AutoEq); see usage
instructions for [using sound signatures](https://github.com/jaakkopasanen/AutoEq#using-sound-signatures) to learn how 
to transfer one headphone into another. In this case the input directory needs to point to the IEM, compensation curve
is the curve of the measurement system used to measure the IEM and the sound signature needs point to the existing
result of the headphone which was used to make the headphone compensation recording.

For example if the headphone compensation recording was made with Sennheiser HD 650 and you want to enjoy Impulcifer
produced HRIR with Campfire Andromeda, you should run:
```bash
python frequency_response.py --input_dir="oratory1990/data/inear/Campfire Audio Andromeda" --output_dir="my_results/Andromeda (HD 650)" --compensation="compensation/harman_over-ear_2018_wo_bass.csv" --sound_signature="results/oratory1990/harman_over-ear_2018/Sennheiser HD 650/Sennheiser HD 650.csv" --equalize --bass_boost=4 --max_gain=6
```
and then copy `AutoEq/my_results/Andromeda (HD 650)/Campfire Audio Andromeda.csv` to `Impulcifer/data/my_hrir/eq.csv`.

See how the Harman over-ear target is used for IEM in this case. This is because the goal is to make Andromeda sound as
similar as possible to HD 650, which is an over-ear headphone. Normally with AutoEQ you would use Harman in-ear target
for IEMs but not in this case.

Headphone compensation can be skipped by adding a command line argument `--no_headphone_compensation` without any value.

#### Sampling Rate
Outputs with different sampling rates than the recording sampling rate can be produced with `--fs` parameter. This
parameter takes a sampling rate in Hertz as value and will then resample the output HRIR to the desired sampling rate if
the recording and output sampling rates differ. For example `--fs=44100`.

#### Plotting Graphs
Various graphs can be produced by providing `--plot` parameter to Impulcifer. These can be helpful in figuring out what
went wrong if the produced HRIR doesn't sound right. Producing the plots will take some time.

- **pre** plots are the unprocessed HRIR measurement
- **room** plots are room measurements done with measurement microphone
- **post** plots are the final results after all processing

#### Channel Balance Correction
Channel balance can be corrected with `--channel_balance` parameter. In ideal case this would not be needed and the
natural channel balance after headphone equalization and room correction would be perfect but this is not always the
case since there are multiple factors which affect that like placement of the binaural microphones. There are six
different strategies available for channel balance correction.

Setting `--channel_balance=trend` will equalize right side by the difference trend of left and right sides. This is a
very smooth difference curve over the entire spectrum. Trend will not affect small deviations and therefore doesn't
warp the frequency response which could lead to uncanny sensations. Bass, mids and treble are all centered when using
trend. Trend is probably the best choice in most situations.

Setting `--channel_balance=mids` will find a gain level for right side which makes the mid frequencies (100, 3000)
average level match that of the left side. This is essentially an automatic guess for the numeric strategy value.

Setting `--channel_balance=1.4` or any numerical value will amplify right side IRs by that number of decibels.
Positive values will boost right side and negative values will attenuate right side. You can find the correct value by
trial and error either with Impulcifer or your virtualization software and running Impulcifer again with the best value.
Typically vocals or speech is good reference for finding the right side gain value because these are most often mixed
in the center and therefore are the most important aspect of channel balance.

Setting `--channel_balance=avg` will equalize both left and right sides to the their average frequency response and
`--channel_balance=min` will equalize them to the minimum of the left and right side frequency response curves. Using
minimum instead of average will be better for avoiding narrow spikes in the equalization curve but which is better in
the end varies case by case. These strategies might cause uncanny sensation because of frequency response warping.

`--channel_balance=left` will equalize right side IRs to have the same frequency response as left side IRs and
`--channel_balance=right` will do the same in reverse. These strategies might cause uncanny sensation because of
frequency response warping.

#### Level Adjustment
Output HRIR level can be adjusted with `--target_level` parameter which will normalize the HRIR gain to the given
numeric value. The level is calculated from all frequencies excluding lowest bass frequencies and highest treble
frequencies and then the level is adjusted to the target level. Setting `--target_level=0` will ensure that HRIR
average gain is about 0 dB. Keep in mind that there often is large variance in the gain of different frequencies so
target level of 0 dB will not mean that the HRIR would not produce clipping. Typically the desired level is several dB
negative such as `--target_level=-12.5`. Target level is a tool for having same level for different HRIRs for easier
comparison.

#### Decay Time Management
The room decay time (reverb time) captured in the binaural room impulse responses can be shortened with `--decay`
parameter. The value is a time it should take for the sound to decay by 60 dB in milliseconds. When the natural decay
time is longer than the given target, the impulse response tails will be shortened with a slope to achieve the desired
decay velocity. Decay times are not increased if the target is longer than the natural one. The decay time management
can be a powerful tool for controlling ringing in the room without having to do any physical room treatments.

## Contact
[Issues](https://github.com/jaakkopasanen/AutoEq/issues) are the way to go if you are experiencing problems, have
ideas or if there is something unclear about how things are done or documented.

You can find me in [Reddit](https://www.reddit.com/user/jaakkopasanen) and
[Head-fi](https://www.head-fi.org/members/jaakkopasanen.491235/) if you just want to say hello.

There is also a [Head-fi thread about Impulcifer](https://www.head-fi.org/threads/recording-impulse-responses-for-speaker-virtualization.890719/).
