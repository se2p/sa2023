# Generated from Expr1.g4 by ANTLR 4.12.0
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,4,22,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,1,0,1,0,1,1,1,1,1,
        2,1,2,1,3,4,3,17,8,3,11,3,12,3,18,1,3,1,3,0,0,4,1,1,3,2,5,3,7,4,
        1,0,1,3,0,9,10,13,13,32,32,22,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,
        0,0,7,1,0,0,0,1,9,1,0,0,0,3,11,1,0,0,0,5,13,1,0,0,0,7,16,1,0,0,0,
        9,10,5,43,0,0,10,2,1,0,0,0,11,12,5,45,0,0,12,4,1,0,0,0,13,14,2,48,
        57,0,14,6,1,0,0,0,15,17,7,0,0,0,16,15,1,0,0,0,17,18,1,0,0,0,18,16,
        1,0,0,0,18,19,1,0,0,0,19,20,1,0,0,0,20,21,6,3,0,0,21,8,1,0,0,0,2,
        0,18,1,6,0,0
    ]

class Expr1Lexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    DIGIT = 3
    WS = 4

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'+'", "'-'" ]

    symbolicNames = [ "<INVALID>",
            "DIGIT", "WS" ]

    ruleNames = [ "T__0", "T__1", "DIGIT", "WS" ]

    grammarFileName = "Expr1.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


