# Reverberation Management
This script crops the reverberant tails of impulse responses in HRIR file.

## Usage
```
usage: reverberation_management.py [-h] --file FILE --track_order TRACK_ORDER
                                   [--reverb REVERB]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Path to HRIR or HeSuVi file.
  --track_order TRACK_ORDER
                        Track order in HRIR file. "hesuvi" or "hexadecagonal"
  --reverb REVERB       Reverberation times for different channels in
                        milliseconds. During this time the reverberation tail
                        will be reduced by 100 dB. A comma separated list of
                        channel name and reverberation time pairs, separated
                        by colon. If only a single numeric value is given, it
                        is used for all channels. When some channel names are
                        give but not all, the missing channels are not
                        affected. Must be at least 3 ms smaller than the HRIR
                        length. For example "--reverb=300" or "--reverb=FL:500
                        ,FC:100,FR:500,SR:700,BR:700,BL:700,SL:700" or "--
                        reverb=FC:100".
```

for example running this from project root:
```bash
python research/reverberation-management/reverberation_management.py --file="data/my_hrir/hesuvi.wav" --track_order=hesuvi --times=FC:300
```
would reduce center channel impulse responses' tails to 300 milliseconds improving speech intelligibility for movies.
