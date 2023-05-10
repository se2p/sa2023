# Generated from SimpleProgramAttributed.g4 by ANTLR 4.12.0
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,15,68,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,3,1,36,8,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        1,2,5,2,47,8,2,10,2,12,2,50,9,2,1,3,1,3,1,3,1,3,3,3,56,8,3,1,4,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,66,8,4,1,4,0,1,4,5,0,2,4,6,8,0,1,1,
        0,8,10,68,0,10,1,0,0,0,2,35,1,0,0,0,4,37,1,0,0,0,6,55,1,0,0,0,8,
        65,1,0,0,0,10,11,3,2,1,0,11,12,6,0,-1,0,12,1,1,0,0,0,13,14,5,13,
        0,0,14,15,5,1,0,0,15,16,3,4,2,0,16,17,6,1,-1,0,17,36,1,0,0,0,18,
        19,5,2,0,0,19,20,3,8,4,0,20,21,5,3,0,0,21,22,6,1,-1,0,22,36,1,0,
        0,0,23,24,5,4,0,0,24,25,3,4,2,0,25,26,5,5,0,0,26,27,3,2,1,0,27,28,
        6,1,-1,0,28,36,1,0,0,0,29,30,5,6,0,0,30,31,3,4,2,0,31,32,5,7,0,0,
        32,33,3,2,1,0,33,34,6,1,-1,0,34,36,1,0,0,0,35,13,1,0,0,0,35,18,1,
        0,0,0,35,23,1,0,0,0,35,29,1,0,0,0,36,3,1,0,0,0,37,38,6,2,-1,0,38,
        39,3,6,3,0,39,40,6,2,-1,0,40,48,1,0,0,0,41,42,10,2,0,0,42,43,7,0,
        0,0,43,44,3,6,3,0,44,45,6,2,-1,0,45,47,1,0,0,0,46,41,1,0,0,0,47,
        50,1,0,0,0,48,46,1,0,0,0,48,49,1,0,0,0,49,5,1,0,0,0,50,48,1,0,0,
        0,51,52,5,12,0,0,52,56,6,3,-1,0,53,54,5,13,0,0,54,56,6,3,-1,0,55,
        51,1,0,0,0,55,53,1,0,0,0,56,7,1,0,0,0,57,58,3,2,1,0,58,59,5,11,0,
        0,59,60,3,8,4,0,60,61,6,4,-1,0,61,66,1,0,0,0,62,63,3,2,1,0,63,64,
        6,4,-1,0,64,66,1,0,0,0,65,57,1,0,0,0,65,62,1,0,0,0,66,9,1,0,0,0,
        4,35,48,55,65
    ]

class SimpleProgramAttributedParser ( Parser ):

    grammarFileName = "SimpleProgramAttributed.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "':='", "'begin'", "'end'", "'if'", "'then'", 
                     "'while'", "'do'", "'+'", "'-'", "'>'", "';'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "Number", "Identifier", "Digit", "WS" ]

    RULE_start = 0
    RULE_statement = 1
    RULE_expr = 2
    RULE_term = 3
    RULE_opt_stmts = 4

    ruleNames =  [ "start", "statement", "expr", "term", "opt_stmts" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    Number=12
    Identifier=13
    Digit=14
    WS=15

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class StartContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.node = None
            self._statement = None # StatementContext

        def statement(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.StatementContext,0)


        def getRuleIndex(self):
            return SimpleProgramAttributedParser.RULE_start

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStart" ):
                listener.enterStart(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStart" ):
                listener.exitStart(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStart" ):
                return visitor.visitStart(self)
            else:
                return visitor.visitChildren(self)




    def start(self):

        localctx = SimpleProgramAttributedParser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 10
            localctx._statement = self.statement()
            localctx.node = localctx._statement.node 
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.node = None
            self._Identifier = None # Token
            self._expr = None # ExprContext
            self._opt_stmts = None # Opt_stmtsContext
            self.a = None # ExprContext
            self._statement = None # StatementContext

        def Identifier(self):
            return self.getToken(SimpleProgramAttributedParser.Identifier, 0)

        def expr(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.ExprContext,0)


        def opt_stmts(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.Opt_stmtsContext,0)


        def statement(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.StatementContext,0)


        def getRuleIndex(self):
            return SimpleProgramAttributedParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = SimpleProgramAttributedParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 35
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13]:
                self.enterOuterAlt(localctx, 1)
                self.state = 13
                localctx._Identifier = self.match(SimpleProgramAttributedParser.Identifier)
                self.state = 14
                self.match(SimpleProgramAttributedParser.T__0)
                self.state = 15
                localctx._expr = self.expr(0)
                localctx.node = AssignmentStatement(Identifier((None if localctx._Identifier is None else localctx._Identifier.text)), localctx._expr.node) 
                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 18
                self.match(SimpleProgramAttributedParser.T__1)
                self.state = 19
                localctx._opt_stmts = self.opt_stmts()
                self.state = 20
                self.match(SimpleProgramAttributedParser.T__2)
                localctx.node = BlockStatement(localctx._opt_stmts.nodes) 
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 3)
                self.state = 23
                self.match(SimpleProgramAttributedParser.T__3)
                self.state = 24
                localctx.a = self.expr(0)
                self.state = 25
                self.match(SimpleProgramAttributedParser.T__4)
                self.state = 26
                localctx._statement = self.statement()
                localctx.node = IfStatement(localctx.a.node, localctx._statement.node) 
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 4)
                self.state = 29
                self.match(SimpleProgramAttributedParser.T__5)
                self.state = 30
                localctx.a = self.expr(0)
                self.state = 31
                self.match(SimpleProgramAttributedParser.T__6)
                self.state = 32
                localctx._statement = self.statement()
                localctx.node = WhileStatement(localctx.a.node, localctx._statement.node) 
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.node = None
            self.a = None # ExprContext
            self._term = None # TermContext
            self.op = None # Token

        def term(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.TermContext,0)


        def expr(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.ExprContext,0)


        def getRuleIndex(self):
            return SimpleProgramAttributedParser.RULE_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpr" ):
                listener.enterExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpr" ):
                listener.exitExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpr" ):
                return visitor.visitExpr(self)
            else:
                return visitor.visitChildren(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = SimpleProgramAttributedParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            localctx._term = self.term()
            localctx.node = localctx._term.node 
            self._ctx.stop = self._input.LT(-1)
            self.state = 48
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = SimpleProgramAttributedParser.ExprContext(self, _parentctx, _parentState)
                    localctx.a = _prevctx
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                    self.state = 41
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 42
                    localctx.op = self._input.LT(1)
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 1792) != 0)):
                        localctx.op = self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 43
                    localctx._term = self.term()
                    localctx.node = Expression(localctx.a.node, localctx._term.node, (None if localctx.op is None else localctx.op.text))  
                self.state = 50
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class TermContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.node = None
            self._Number = None # Token
            self._Identifier = None # Token

        def Number(self):
            return self.getToken(SimpleProgramAttributedParser.Number, 0)

        def Identifier(self):
            return self.getToken(SimpleProgramAttributedParser.Identifier, 0)

        def getRuleIndex(self):
            return SimpleProgramAttributedParser.RULE_term

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTerm" ):
                listener.enterTerm(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTerm" ):
                listener.exitTerm(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTerm" ):
                return visitor.visitTerm(self)
            else:
                return visitor.visitChildren(self)




    def term(self):

        localctx = SimpleProgramAttributedParser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_term)
        try:
            self.state = 55
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [12]:
                self.enterOuterAlt(localctx, 1)
                self.state = 51
                localctx._Number = self.match(SimpleProgramAttributedParser.Number)
                localctx.node = Number((None if localctx._Number is None else localctx._Number.text)) 
                pass
            elif token in [13]:
                self.enterOuterAlt(localctx, 2)
                self.state = 53
                localctx._Identifier = self.match(SimpleProgramAttributedParser.Identifier)
                localctx.node = Identifier((None if localctx._Identifier is None else localctx._Identifier.text)) 
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Opt_stmtsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.nodes = None
            self._statement = None # StatementContext
            self._opt_stmts = None # Opt_stmtsContext

        def statement(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.StatementContext,0)


        def opt_stmts(self):
            return self.getTypedRuleContext(SimpleProgramAttributedParser.Opt_stmtsContext,0)


        def getRuleIndex(self):
            return SimpleProgramAttributedParser.RULE_opt_stmts

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOpt_stmts" ):
                listener.enterOpt_stmts(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOpt_stmts" ):
                listener.exitOpt_stmts(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpt_stmts" ):
                return visitor.visitOpt_stmts(self)
            else:
                return visitor.visitChildren(self)




    def opt_stmts(self):

        localctx = SimpleProgramAttributedParser.Opt_stmtsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_opt_stmts)
        try:
            self.state = 65
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 57
                localctx._statement = self.statement()
                self.state = 58
                self.match(SimpleProgramAttributedParser.T__10)
                self.state = 59
                localctx._opt_stmts = self.opt_stmts()
                localctx.nodes = [ localctx._statement.node] + localctx._opt_stmts.nodes 
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 62
                localctx._statement = self.statement()
                localctx.nodes = [ localctx._statement.node] 
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[2] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 2)
         




