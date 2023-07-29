class BadSignalWarning(Exception):
    def __init__(self):
        self.msg = "Bad signal warning"
        super().__init__(self.msg)


class ShapeNotCorrect(Exception):
    def __init__(self, shape=None):
        if shape:
            self.msg = f"Shape not correct, shape is {shape}"
        else:
            self.msg = "Shape not correct"
        super().__init__(self.msg)


class AbnormalBPM(Exception):
    def __init__(self, bpm):
        if bpm:
            self.msg = "fabnormal bpm, bpm is {bpm}"
        else:
            self.msg = "abnormal bpm"
        super().__init__(self.msg)

class EntropyError(Exception):
    def __init__(self):
        self.msg = "Entropy calculate error"
        super().__init__(self.msg)

class AccStatsCalcError(Exception):
    def __init__(self):
        self.msg = "AccStatsCalcError"
        super().__init__(self.msg)

class ParsingError(Exception):
    def __init__(self):
        self.msg = "Parsing result error"
        super().__init__(self.msg)


class NoOutputError(Exception):
    def __init__(self):
        self.msg = "NoOutputError"
        super().__init__(self.msg)
