
BPM = 120

DURATION_QUARTER_NOTE = 1

DURATION_EIGHTH_NOTE = DURATION_QUARTER_NOTE/2
DURATION_SIXTEENTH_NOTE = DURATION_EIGHTH_NOTE/2

DURATION_HALF_NOTE = 2 * DURATION_QUARTER_NOTE
DURATION_WHOLE_NOTE = 2 * DURATION_HALF_NOTE

durations = {
    'sixteenth':DURATION_EIGHTH_NOTE,
    'eighth':DURATION_EIGHTH_NOTE,
    'quarter':DURATION_QUARTER_NOTE,
    'half':DURATION_HALF_NOTE,
    'whole':DURATION_WHOLE_NOTE
}

class Symbol(object):
    @staticmethod
    def get_symbol(name):
        if 'note' in name:
            tmp = name.split('.')
            duration = durations[tmp[0][:-5]]
            n = int(tmp[1])
            return Note(n, duration)
        
        if name in ['sharp', 'flat', 'neutral']:
            return Accidental(name)
        
        if 'clef' in name:
            return Clef(name[:-5])
        
        if 'rest' in name:
            duration = durations[name[:-5]]
            return Rest(duration)
        
        if name == 'dot':
            return Dot()
    
    def action(self, track_builder):
        pass

class Note(Symbol):
    def __init__(self, n, duration):
        self.n = n
        self.duration = duration
    
    def action(self, track_builder):
        track_builder.add_note(self)

class Clef(Symbol):
    def __init__(self, name):
        self.name = name
    
    def action(self, track_builder):
        track_builder.set_clef(self)

class Accidental(Symbol):
    def __init__(self, name):
        self.name = name
    
    def action(self, track_builder):
        track_builder.add_accidental(self)

class Rest(Symbol):
    def __init__(self, duration):
        self.duration = duration
    
    def action(self, track_builder):
        track_builder.add_rest(self)
