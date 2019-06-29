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
- Download [AutoEQ zip](https://github.com/jaakkopasanen/Impulcifer/archive/master.zip) and exctract to a convenient
location. Or just git clone if you know what that means.
- Download and install [Python3](https://www.python.org/getit/). Make sure to check
*Install Python3 to PATH*
- Install virtualenv. Run this on command prompt. Search `cmd` in Windows start menu.  
```bash
pip install virtualenv
```
- Go to Impulcifer folder  
```bash
cd C:\path\to\Impulcifer-master
```
- Create virtual environment  
```bash
virtualenv venv
```
- Activate virtualenv  
```bash
venv\Scripts\activate
```
- Install required packages  
```bash
pip install -r requirements.txt
```
- Verify installation  
```bash
python impulcifer.py -H
```

When coming back at a later time you'll only need to activate virtual environment again
```bash
cd C:\path\to\Impulcifer-master
venv\Scripts\activate
```

## Demo
The actual HRIR measurements require a little investment in measurement gear and the chances are that you're here before
you have acquired them. There is a demo available for testing out Impulcifer without having to do the actual
measurements. `data\demo` folder contains two measurement files which are needed for running Impulcifer. `recording.wav`
has the sine sweep measurements done with speakers and `headphones.wav` has the same done with headphones.

You can try out what Impulcifer does by running:
```bash
python impulcifer.py --speakers=FL,FR --dir_path=data/demo --test_signal=data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav --compensate_headphones
```
Impulcifer will now process the measurements and produce `hrir.wav` and `hesuvi.wav` which can be used with headphone
speaker virtualization software such as [HeSuVi](https://sourceforge.net/projects/hesuvi/) to make headphones sound like
speakers in a room. When testing with HeSuVi copy `hesuvi.wav` into `C:\Program Files\Equalizer APO\config\Hesuvi\hrir`,
(re)start HeSuVi and select `hesuvi.wav` from the Common HRIRs list on Virtualization tab.

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

![The Sound Professionals binaural microphones](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/binaural_mics.png)

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

Low noise is very desirable for the binaural microphones because lower noise will means better signal to noise ratio
for the HRIR. All of these mics are electet microphones are as such require bias voltage (plug-in power) between 5 to
12 volts. These are typically designed to be used with digital recorders. The Sound Professionals mics with XLR
connectors can be used with normal USB audio interfaces which offer significantly lower equivalent input noise
performance than digital recorders. USB audio interfaces don't normally have plug-in power and therefore cannot be used
with electret mics. XLR versions of The Sound Professionals mics have adapters which convert 48 volts of phantom power
to 9 or 12 volts of plug-in power for the capsules.

Microphones are placed as close to ear canal opening as possible. Having the mics too far away from the ear canal could
affect how the outer ear affects the sound and having the mics crammed too tightly into the ear canal could have huge
dampening effect on the high frequencies.

![The Sound Professionals binaural microphones](https://raw.githubusercontent.com/jaakkopasanen/Impulcifer/master/img/mic_placement.png)

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

### Sound Device Setup
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

#### Recording with Audacity
- Settings
    - Custom output channel mapping
   - Over-dub recording
- Open sweep sequence
- Headphones
    - Done first in case microphones move
    - Put on headphones and select headphones output
    - Disable all processing
    - Set volume to high comfortable level
    - Record over-dub with headphones
    - Select recorded stereo track -> File -> Export -> Export selected audio -> Save to directory as 32-bit WAV
- Speakers
    - Take headphones off
    - Disable all processing except room correction
    - Set volume to high comfortable level
    - Record over-dub with speakers
    - Adjust microphone input volume if level is not good and record again
    - Select recorded stereo track -> File -> Export -> Export selected audio -> Save to directory as 32-bit WAV


Measurement gear and procedure goes here...

## Usage
Usage instructions and command line arguments go here...

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