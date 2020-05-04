from mingus.containers.bar import Bar
from mingus.containers.track import Track
from mingus.containers.note import Note as mingusNote
from mingus.containers.composition import Composition
from mingus.midi import fluidsynth
from symbols import Symbol

class TrackBuilder(object):
    def __init__(self):
        self.track = Track()
        self.previous = None

    def add_symbol(self, name):
        symbol = Symbol.get_symbol(name)
        self.previous = symbol
        self.accidental = 0
        symbol.action(self)

    def add_note(self, note):
        mnote = mingusNote(note.n)
        self.track.add_notes(mnote, note.duration)
    
    def add_rest(self, rest):
        self.track.add_notes(None, rest.duration)

    def set_clef(self, clef):
        self.clef = clef

    def add_accidental(self, accidental):
        if accidental.name == 'sharp':
            self.accidental = 1
        elif accidental.name == 'flat':
            self.accidental = -1
        else:
            self.accidental = 0

    def flush(self):
        return self.track
    
def main():
    comp = Composition()
    with open('out.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tb = TrackBuilder()
            symbols = line.rstrip().split(' ')[1:]
            print(symbols)
            for symbol in symbols:
                tb.add_symbol(symbol)
            track = tb.flush()
            comp.add_track(track)
    fluidsynth.init('../sfs/soundfont.sf2', 'alsa')
    fluidsynth.play_Composition(comp)

    

if __name__ == "__main__":
    main()