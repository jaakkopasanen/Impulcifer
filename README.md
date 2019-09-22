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
acoustic corrections which are normally possible in real rooms with DSP.

## Installing
Impulcifer is used from a command line and requires some prerequisites. These installation instructions will guide you
through installing everything that is needed to run Impulcifer on you own PC.

- Download and install Git: https://git-scm.com/downloads. When installing Git on Windows, use Windows SSL verification
instead of Open SSL or you might run into problems when installing project dependencies.
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
# On Windows
venv/Scripts/activate
# On Mac and Linux
source venv/bin/activate
```
- Install required packages.  
```bash
pip install -r requirements.txt
```
- Verify installation. You should see help printed if everything went well.  
```bash
python impulcifer.py --help
```

When coming back at a later time you'll only need to activate virtual environment again before using Impulcifer.
```bash
cd Implucifer
# On Windows
venv/Scripts/activate
# On Mac and Linux
source venv/bin/activate
```

Impulcifer is under active development and updates quite frequently. You can update your own copy by running:
```bash
git pull
```

## Demo
The actual HRIR measurements require a little investment in measurement gear and the chances are that you're here before
you have acquired them. There is a demo available for testing out Impulcifer without having to do the actual
measurements. `data/demo` folder contains five measurement files which are needed for running Impulcifer.
`headphones.wav` has the sine sweep recordings done with headphones and all the rest files are recordings done with
stereo speakers in multiple stages.

You can try out what Impulcifer does by running:
```bash
python impulcifer.py --dir_path=data/demo --test_signal=data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav
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
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/headphones.wav"
```
- Take heaphones off, look forward and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FC,FR,SR,BR,BL,SL-7.1-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FL,FC,FR,SR,BR,BL,SL.wav"
```
- Process recordings  
```bash
python impulcifer.py --dir_path="data/my_hrir" --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav"
```

### Stereo Speaker Setup
Steps and commands for doing measurements with two speakers in four stages:
- Put microphones in ears, put headphones on and run
```bash
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/headphones.wav"
```
- Take heaphones off, look forward and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FL,FR.wav"
```
- Look 120 degrees left (left speaker should be on your right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/SR,BR.wav"
```
- Look 120 degrees right (right speaker should be on your left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/BL,SL.wav"
```
- Look directly at the left (or right) speaker and run either one of these commands  
```bash
# Using left speaker 
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FC.wav"
# Using right speaker 
python recorder.py --play="data/sweep-seg-FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FC.wav"
```
- Process recordings  
```bash
python impulcifer.py --dir_path="data/my_hrir" --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav"
```

### Single Speaker
Steps and command for doing measurements with just a single speaker in 7 steps. All speaker measurements use either
`sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` or
`sweep-seg-FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` depending if the speaker is connected to left or right
cable terminals in the amplifier. This commands assume the speaker is connected to left speaker terminals.
- Put microphones in ears, put headphones on and run
```bash
python recorder.py --play="data/sweep-seg-FL,FR,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/headphones.wav"
```
- Look 30 degrees right of the speaker (speaker on your front left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FL.wav"
```
- Look directly at the speaker and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FC.wav"
```
- Look 30 degrees left of the speaker (speaker on you front right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/FR.wav"
```
- Look 90 degrees left of the speaker (speaker on your right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/SR.wav"
```
- Look 150 degrees left of the speaker (speaker on your back right) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/BR.wav"
```
- Look 150 degrees right of the speaker (speaker on you back left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/BL.wav"
```
- Look 90 degrees right of the speaker (speaker on your left) and run  
```bash
python recorder.py --play="data/sweep-seg-FL,stereo-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --record="my_hrir/SL.wav"
```
- Process recordings  
```bash
python impulcifer.py --dir_path="data/my_hrir" --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav"
```

## Processing
Once you have obtained the sine sweep recordings, you can turn them into HRIR file with Impulcifer. All the processing
is done by running a single command on command line. The command below assumes you have made a speaker recording and
a headphones recording and saved the recording files into `data/my_hrir` folder. Start command prompt, jump to
Impulcifer folder and activate the virtual environment as described in the installation instructions if you don't have
command prompt open yet. Sine sweep recordings are processed by running `impulcifer.py` with Python as shown below.
```bash
python impulcifer.py --dir_path="data/my_hrir" --test_signal="data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav" --plot
```

You should have several WAV files and graphs in the folder. `hesuvi.wav` can now be used with HeSuVi to make your
headphones sound like speakers. Let's see what these arguments mean.

`--dir_path=data/my_hrir` tells Impulcifer that the recordings can be found in a folder called `my_hrir` under `data`.
Impulcifer will also write all the output files into this folder.

`--test_signal=data/sweep-6.15s-48000Hz-32bit-2.93Hz-24000Hz.wav` tells Impulcifer that the sine sweep signal is this
WAV file. Impulcifer will load the file and construct inverse filter from the test signal and use that inverse filter
to turn sine sweep recordings into impulse responses. Seconds, bits, and Hertzs in the file name are actually not
important and the file name can be anything. `--test` argument doesn't need to be supplied if the folder contains a file
called `test.wav`.

Impulcifer will compensate for the headphone frequency response using headphone sine sweep recording if the folder
contains file called `headphones.wav`. If you have the file but would like not to have headphone compensation, you can
simply rename the file for example as `headphones.wav.bak` and run the command again.

Various graphs can be produced by providing `--plot` parameter to Impulcifer. These can be helpful in figuring out what
went wrong if the produced HRIR doesn't sound right. Producing the plots will take some time so be patient.

## Contact
[Issues](https://github.com/jaakkopasanen/AutoEq/issues) are the way to go if you are experiencing problems, have
ideas or if there is something unclear about how things are done or documented.

You can find me in [Reddit](https://www.reddit.com/user/jaakkopasanen) and
[Head-fi](https://www.head-fi.org/members/jaakkopasanen.491235/) if you just want to say hello.

There is also a [Head-fi thread about Impulcifer](https://www.head-fi.org/threads/recording-impulse-responses-for-speaker-virtualization.890719/).
