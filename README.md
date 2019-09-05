# Impulcifer
Impulcifer is a tool for creating head related impulse responses (HRIR) for speaker virtualization on headphones.

Normally headphones sound inside your head which is a clear disadvantage for games and movies but also for music
because basically all material has been created for speakers. Virtual surround technologies for headphones have existed
for a some time by now but almost all of them fail to fulfill expectations of out of head sound localization and the
naturalness of speakers. This is because your brains have learned to localize sounds only with your ears and head not
not with anyone elses. Surround sound on headphones can only be convincing when the surround virtualization technology
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
acoustic corrections which are normally possible in real rooms with DSP.

## Installing
- Download and install Git: https://git-scm.com/downloads
- Download and install [Python3](https://www.python.org/getit/). Make sure to check
*Add Python 3 to PATH*

Rest will be done in terminal / command prompt. On Windows you'll find it by searching `cmd` in Start menu.
You should be able to simply copy and paste in these commands. 

- Install virtualenv.  
```bash
pip install virtualenv
```
- Git clone Impulcifer. This will create a folder in your home folder called `Impulcifer`.  
```bash
git clone https://github.com/jaakkopasanen/Impulcifer
```
- Go to Impulcifer folder.  
```bash
cd Implucifer
```
- Create virtual environment for the project.  
```bash
virtualenv venv
```
- Activate virtualenv.  
```bash
venv\Scripts\activate
```
- Install required packages.  
```bash
pip install -r requirements.txt
```
- Verify installation. You should see help printed if everything went well.  
```bash
python impulcifer.py -H
```

When coming back at a later time you'll only need to activate virtual environment again before using Impulcifer.
```bash
cd Implucifer
venv\Scripts\activate
```

## Demo
The actual HRIR measurements require a little investment in measurement gear and the chances are that you're here before
you have acquired them. There is a demo available for testing out Impulcifer without having to do the actual
measurements. `data\demo` folder contains two measurement files which are needed for running Impulcifer. `recording.wav`
has the sine sweep measurements done with speakers and `headphones.wav` has the same done with headphones.

You can try out what Impulcifer does by running:
```bash
python impulcifer.py --dir_path=data/demo --speakers=FL,FR,SR,BR,BL,SL,X,FC --test_signal=data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav --compensate_headphones
```
Impulcifer will now process the measurements and produce `hrir.wav` and `hesuvi.wav` which can be used with headphone
speaker virtualization software such as [HeSuVi](https://sourceforge.net/projects/hesuvi/) to make headphones sound like
speakers in a room. When testing with HeSuVi copy `hesuvi.wav` into `C:\Program Files\Equalizer APO\config\Hesuvi\hrir`,
(re)start HeSuVi and select `hesuvi.wav` from the Common HRIRs list on Virtualization tab.

Demo recording has 8 speakers recorded with two physical speakers. First front left and front right speakers were recorded on left and right speaker while looking directly in the middle of the speakers. Second iteration recorded side right and back right speakers on left and right speaker while looking 120 degrees left. When looking 120 degrees left, the left speaker which is normally 30 degrees to left is now 90 degrees to right and right speaker which is normally 30 degrees to right is now 150 degrees to right. 90 degrees right and 150 degrees right correspond to standard side right (SL) and back right (BR) speakers. Same trick was done for back left and side left channels but this time recording while looking at 120 degrees right. When looking at 120 degrees right, left speaker is at 150 degrees left and right speaker at 90 degrees left. Lastly center speaker was recorded while looking directly at right speaker. When looking at right speaker the left speaker is at 60 degrees to left which does not match any standard surround speakers. That's why the speaker sequence has X at the 7th element. This tells Impulcifer to simply ignore it.

## Measurement
HRIR measurements are done with binaural microphones which are also called ear canal blocking microphones or in-ear
microphones. Exponential sine sweep test signal is played on speakers and the sound is recorded with the microphones at
ear canal opening. This setup ensures that the sound measured by the microphones is affected by the room and your body,
head and ears just like it is when listening to music playing on speakers. Impulcifer will then transform these
recordings into impulse responses, one per each speaker and ear pair.

### Measurement Gear
Binaural microphones, audio interface, headphones and speakers are required to measure HRIR.

#### Microphones
Microphones used for HRIR measurements need to be small enough to fit inside your ears and have some kind of plug or
hook for keeping the microphone capsule in place during the measurements.
[The Sound Professionals binaural microphones](https://www.soundprofessionals.com/cgi-bin/gold/item/SP-TFB-2)
have been proven to work well for the task. They have a small electret capsule and a plastic hook which helps securing
the microphones. Pictured below is the basic model with cable taped to a loop in order to get the cable exit ears from
front and help further hold the microphones in place.

Microphones are placed as close to ear canal opening as possible. Having the mics too far away from the ear canal could
affect how the outer ear affects the sound and having the mics crammed too tightly into the ear canal could have huge
dampening effect on the high frequencies.

![Binaural microphones](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/mics.jpg)

Some commercially available binaural microphones are:
- [The Sound Professionals SP-TFB-2](https://www.soundprofessionals.com/cgi-bin/gold/item/SP-TFB-2) with 36 dB noise and -30 dB sensitivity
- [The Sound Professionals SP-TFB-2 with XLR](https://www.soundprofessionals.com/cgi-bin/gold/item/SP-TFB-2-BLK-HS-MOG-XLR)
- [The Sound Professionals MS-TFB-2](https://www.soundprofessionals.com/cgi-bin/gold/item/MS-TFB-2) with 19 dB noise and -32 dB sensitivity
- [The Sound Professionals MS-TFB-2 with XLR](https://www.soundprofessionals.com/cgi-bin/gold/item/MS-TFB-2-MOG-XLR)
- [Roland CS-10EM](https://www.roland.com/us/products/cs-10em/) with < 34 dB noise and -40 dB sensitivity
- [Microphone Madness MM-BS-8](https://www.microphonemadness.com/mm-bsm-8-true-natural-over-the-ear-in-ear-binaural-stereo-microphones.html) with 32 dB equivalent noise and -35 db sensitivity
- [Soundman binaural microphone line-up](http://www.soundman.de/en/products/)
- [Core Sound binaural microphones](http://www.core-sound.com/mics/1.php) with 30 dB noise
- [Core Sound high-end binaural microphones](http://www.core-sound.com/bk/1.php) with 28 dB noise for 4060 capsule and 23 dB noise for 4061 capsule

Low noise is very desirable for the binaural microphones because lower noise means better signal to noise ratio
for the HRIR. All of these mics are electet microphones are as such require bias voltage (plug-in power) between 5 to
12 volts. These are typically designed to be used with digital recorders. The Sound Professionals mics with XLR
connectors can be used with normal USB audio interfaces which offer significantly lower equivalent input noise
performance than digital recorders. USB audio interfaces don't normally have plug-in power and therefore cannot be used
with electret mics. XLR versions of The Sound Professionals mics have adapters which convert 48 volts of phantom power
to 9 or 12 volts of plug-in power for the capsules.

#### Audio Interface
Audio interface is the microphone input for the PC. As mentioned above most USB audio interfaces don't have plug-in
power for electret mics and cannot be used with them without adapters. It is possible to buy or build an adapter which
converts phantom power from XLR to plug-in power on 3.5 mm stereo jack or use mics which have this built in. Excellent
noise performance can be had very cheaply with [Behringer UMC202HD](https://www.youtube.com/watch?v=YquAMkHJYjA)
but other popular options such as [Focusrite 2i2 2.Gen](https://www.youtube.com/watch?v=FdH5cnjmfUc) should work just as
well.

Alternative to normal USB audio interfaces is digital recorders which can function as a USB audio interface. Most
digital recorders cannot so the options here are limited. One suitable option is
[Zoom H1n](https://www.zoom-na.com/products/field-video-recording/field-recording/zoom-h1n-handy-recorder) although
optimal results cannot be guaranteed since H1n has a whopping 20 dB higher input noise than Behringer UMC202HD.

Finally it is possible to measuring HRIRs with digital stereo recorders which cannot act as USB audio interface but in
this case you have to play the sine sweep on PC, use the recorder without being connected to PC and then manually
transfer the recorded files from the recorder's SD card to PC.

#### Speakers and Headphones
In theory any decent speakers and around ear headphones work but the end result will depend on the performance of the
speakers and headphones.

Low harmonic distortion is wanted for the speakers because measurements should be done with
relatively high volume for better signal to noise ratio. Exponential sine sweep measurement method used by Impulcifer
cancels out most of the harmonic distortion of the speakers but some residual distortion may remain. Frequency response
and transient response is not so important because Impulcifer can do room correction better than what is physically
possible with real speakers and sine sweeps don't have transients at all so that aspect of the speakers isn't modeled by
HRIR. All in all speakers don't have to be expensive high-end devices but any decent affordable HiFi speaker should do
the job sufficiently.

Headphones will have major impact on the speaker virtualization quality. Frequency response, which is normally by far
the most important aspect of headphones' sound quality, isn't very critical in this application because it will be
equalized as a part of the HRIR measurement process. Headphones' other qualities will have direct impact on the end
result and therefore fast and well resolving headphones are recommended although any decent pair of open back headphones
can create illusion of listening to speakers in a real room. Some headphones are better suited for for speaker
virtualization but it's quite not known at this time which elements affect the localization, externalization and
plausability of binaural reproduction. Electrostatic headphones and Sennheiser HD 800 at least have proven themselves as
reliable tools for binaural use.

#### Sound Device Setup
Input and output devices in Windows (or whatever OS you are on) need to be configured for the measurement process. One
measurement is done with speakers and one with headphones so if you have different output devices (soundcard) for them
then you need to configure both. For the sake of simplicity we will here go through the setup for using the audio
interface for speaker output, headphone output and microphone input.

Go to Windows sound settings: Control panel -> Hardware and Sound -> Sound. Select your output device (audio interface),
click Set Default and then click properties. Go to Advanced tab and select a format with highest possible bit number and
48000 Hz. Other sampling frequencies are possible but 48000 Hz is default on HeSuVi and covers all use cases. Click OK
to close the output device properties.
![Output device](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/output_device.png)

Next go to Recording tab on Windows sound settings, select your input device (audio interface), set it as default and
select same format from the Properties as you selected for the output device.
![Output devices](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/input_device.png)

### Recording with Audacity
Audacity is a free audio workstation and while being simpler than it's commercial counterparts it has all the features
needed for recording sine sweep measurements. Download and install Audacity from
[Audacity's website](https://www.audacityteam.org/download/).

Audacity can do overdub recordings meaning that Audacity will play back the existing tracks at the same time as it is
recording new tracks. Overdub recordings need to be enabled in the options for this. Go to Edit -> Preferences ->
Recording and select "Play other tracks while recording (overdub)" and "Record on a new track". Overdub recordings
aren't strictly necessary but make the process easier when playing the sine sweeps on one or two speakers.

When saving the recording as WAV file Audacity needs to create multi-track file. This too needs to be enabled from the
options. Go to Edit -> Preferences -> Import / Export and select "Use custom mix".
![Audacity settings](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/audacity_settings.png)

Once you have the settings configured you can start recording. Open up exponential sine sweep sequence file from
`data\sweep-seg-FL,FR-stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` in audacity. We'll record on two speakers in this
guide but it is possible to record even full 7 channel surround sound setup with just one speaker.
![Sweep sequence in Audacity](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/sweep_sequence.png)

Input and output devices should be correctly selected if you set them as default devices in Windows Sound Settings but
check once more in the tool bar that input and output device is correct (Zoom H1n in this case) and "2 (Stereo)
Recording" is selected as the recording format for the input device.

Now you're ready to start recording. Headphones should be measured first because putting the headphones on and taking
them off might move the microphones if they are not secured well and while this isn't necessarily catastrophic we'll
want the microphones to be in the same position for both the headphone measurement and the speaker measurement.

Put on your headphones and set the output volume to a comfortable high level. High volume is better because it ensures
higher signal to noise ratio but the volume shouldn't be so high that listening to the sine sweep is uncomfortable. Also
too high volume might cause significant distortion on certain headphones and speakers. If you have any existing audio
processing going on, you should disable them now.

Click the big red record button to start recording. Single ascending frequency should start playing after two seconds of
silence. End the recording with stop button after the entire sweep sequence has played. Recording has to be longer than
the sweep sequence.
![Headphones Recording](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/headphones_recording.png)

The waveform should reach close to maximum but should never touch it. If there are samples that are at the maximum then
most likely they went over it and clipped. Clipping causes massive distortion and will ruin the measurement. If the
highest point of the waveform is close to maximum (or lowest is close to minimum) you can check if they are in fact
within the limits by selecting the recorded track, opening Amplify tool from Effect -> Amplify, setting "New Peak
Amplitude" to 0.0 dB and looking at the "Amplification (dB)" value. If the value is above zero dB you are safe.
![Audacity Amplify Tool](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/audacity_amplify.png)

Often the level is not going to be within optimal limits the first try and if this is the case you need to adjust mic
input gain. Typically the audio interface has a physical knob for the microphone gain. If the recorded track is very
low on volume, increase the mic gain and reduce the gain if the waveform clipped. Run the recording again after
adjusting the mic gain and remove the old track from the small x button on the top left corner of the track.

When you have successfully recorded the frequency sweep with headphones you need to save the stereo track to a WAV file.
Select the track if not selected already and go to File -> Export -> Export Selected Audio. Create a new folder inside
`Impulcifer-master\data` called `my_hrir` and open the folder. Name the file as `headphones.wav` select file type as
"WAV (Microsoft) 32-bit float PCM" and click Save. Audacity will ask you about the mix, if you selected custom mix from
the settings earlier, and you should select the two channel output where first track is mapped to the first output
channel and the second track to second output channel.
![Audacity WAV Export](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/audacity_wav_export.png)

Speaker recording for a stereo setup goes exactly like the headphone recording. Unplug your headphones and set the
volume on speakers to a comfortable high level. Only the original sweep sequence must be playing and to prevent
Audacity from playing the headphone recording as well you need to mute it. Click the "Mute" button on the headphone
recording track control panel below the delete button. Track turns grey.

Start recording with the same red record button, wait silently and without moving until the sweep sequence has played
entirely and stop the recording with the stop button.
![Speakers Recording](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/speakers_recording.png)

Check the speaker recording levels and export to a file called `recording.wav` inside the same folder. Congratulations,
you have your first HRIR measurement finished. Now you need to process the sine sweep recordings into head related
impulse responses with Impulcifer.

## Usage
Start command prompt and jump to Impulcifer-master folder and activate the virtual environment as described in the
installation instructions. Sine sweep recordings are processed by running `impulcifer.py` with Python as shown in the
demo section.

Processing the recordings made above and saved to `my_hrir` folder can be done by running:
```bash
python impulcifer.py --dir_path=data\my_hrir --test_signal=data\sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav --speakers=FL,FR --compensate_headphones
```

You should have several WAV files and graphs in the folder. `hesuvi.wav` can now be used with HeSuVi to make your
headphones sound like speakers. Let's see what these arguments mean.

`--dir_path=data\my_hrir` tells Impulcifer that the recordings can be found in a folder called `my_hrir` under `data`.
Impulcifer will also write all the output files into this folder.

`--test_signal=data\sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` tells Impulcifer that the sine sweep signal is this
WAV file. Impulcifer will load the file and construct inverse filter from the test signal and use that inverse filter
to turn sine sweep recordings into impulse responses. Seconds, bits, and Hertzs in the file name are actually not
important and the file name can be anything. `--test` argument doesn't need to be supplied if the folder contains a file
called `test.wav`.

`--speakers=FL,FR` tells Impulcifer that speaker recording was performed on front
left and front right speakers in that order. This could be for example `FL,FC,FR,SR,BR,BL,SL` when recording 7 channel
surround setup.

`--compensate_headphones` requests Impulcifer to compensate for the headphone frequency response using headphone sine
sweep recording.

## Algorithms
- Overview
- Impulse response estimation
  - Pre-ringing cancellation
- Tracking filter
- Channel delay syncing
  - Channel delay calculation (equal distance to point between ears)
- Reverberation time adjustment
- Early reflection management
- Room correction
- Headphone compensation
- Plots

### Headphone Compensation
Headphones and measurement microphones have a frequency responses of their own and these will have to be compensated in
order to have real HRIR and virtualized HRIR sound the same. Frequency response of headphones and measurement
microphones can be compensated by measuring frequency response of headphones with the measurement microphones in ears
and equalizing the measured frequency response flat. This way the frequency response at the ear canal opening will be
the same when listening to the speakers or virtualizing the speakers on headphones.

Impulcifer can do this headphone compensation as long as the data directory contains exponential sine sweep measurement
file called `headphones.wav`. Impulcifer will calculate the frequency response of headphones-microphones system from the
sine sweep recording and creates a equalization FIR filters by inverting the frequency responses of left and right ear.
All left ear impulse responses of the measured HRIR are then equalized with the left side equalization FIR filter and
right ear impulse responses with the right side equalization FIR filter. Impulcifer will take the absolute gain values
of headphone-microphone frequency responses into account so any channel imbalances will be compensated as well.

Because the headphone compensation recording is done with the same binaural measurement micrphones the equalization
filters will equalize the microphone frequency responses too. This creates the benefit of not needing to have
microphones with flat frequency response.

Script `compensation.py` and recordings at `data/compensation` serve as a proof that headphone compensation works as
intended. `data/compensation` contains three recordings: normal HRIR recording, heaphone sine sweep recording and
virtualized HRIR recording. Virtualized HRIR is done on headphones with virtualization processing using HRIR created
with the first two recordings. Graphs below show that the frequency responses of raw HRIR and virtualized HRIR match
each other very closely. Narrow notches in the error curve are caused by smart smoothing which avoids narrow positive
gain spikes in the equalization filters.

![Headphone compensation graphs](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/data/headphone_compensation/headphone_compensation.png)

This compensation scheme is based on a naive assumption that if a frequency response is the same at the ear
canal opening when listening to speakers versus when listening virtualized speakers on headphones then the frequency
responses will be the same at the ear drum. In truth this assumption does not hold true for most headphones. Even large
around ear headphones will affect ear canal resonances changing the frequency response at the ear drum. Unfortunately
it is not possible to measure headphone's effect of ear canal resonances with binaural microphones that sit at the ear
canal entrance.
