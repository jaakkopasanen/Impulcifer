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
- Go to AutoEQ location  
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
python frequency_response.py -H
```

When coming back at a later time you'll only need to activate virtual environment again
```bash
cd C:\path\to\Impulcifer-master
venv\Scripts\activate
```

## Demo
The actual HRIR measurements require a little investment in measurement gear and the chances are that you're here before
you have acquired them. There is a demo available for testing out Impulcifer without having to do the actual
measurements. `data/demo` folder contains two measurement files which are needed for running Impulcifer. `recording.wav`
has the frequency sweep measurements done with speakers and `headphones.wav` has the same done with headphones.

You can try out what Impulcifer does by running:
```bash
python impulcifer.py --silence_length=2 --speakers=FL,FR --dir_path=data/demo --test_signal=data/Sweep-48000-32bit-1ch-5s-20Hz-24000Hz.wav --compensate_headphones
```
Impulcifer will now process the measurements and produce `hrir.wav` and `hesuvi.wav` which can be used with headphone
speaker virtualization software such as [HeSuVi](https://sourceforge.net/projects/hesuvi/) to make headphones sound like
speakers in a room. When testing with HeSuVi copy `hesuvi.wav` into `C:\Program Files\Equalizer APO\config\Hesuvi\hrir`,
(re)start HeSuVi and select `hesuvi.wav` from the Common HRIRs list on Virtualization tab.

## Measurement
HRIR measurements are done with binaural microphones which are also called ear canal blocking microphones or in-ear
microphones. Logarithmic sine sweep test signal is played on speakers and the sound is recorded with the microphones at
ear canal opening. This setup ensures that the sound measured by the microphones is affected by the room and your body,
head and ears just like it is when listening to music playing on speakers. Impulcifer will then transform these
recordings into impulse responses, one per each speaker and ear pair.

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