# -*- coding: utf-8 -*-

# https://en.wikipedia.org/wiki/Surround_sound
SPEAKER_NAMES = ['FL', 'FR', 'FC', 'BL', 'BR', 'SL', 'SR']

# Each channel, left and right
IR_ORDER = []
for _ch in SPEAKER_NAMES:
    IR_ORDER.append(_ch+'-left')
    IR_ORDER.append(_ch+'-right')

# Time delays between speaker to primary ear vs speaker to middle of head in milliseconds
# Speaker configuration is perfect circle around listening position
# Distance from side left speaker (SL) to left ear is smaller than distance from front left (FL) to left ear
SPEAKER_DELAYS = {
    'FL': 0.107,
    'FR': 0.107,
    'FC': 0.214,
    'BL': 0.107,
    'BR': 0.107,
    'SL': 0.0,
    'SR': 0.0,
}