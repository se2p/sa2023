# Generated from Expr4.g4 by ANTLR 4.12.0
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
        4,1,15,77,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,0,1,
        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,39,8,1,1,2,1,2,1,2,1,2,1,2,
        1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,5,2,59,8,2,10,
        2,12,2,62,9,2,1,3,1,3,1,3,1,3,3,3,68,8,3,1,4,1,4,1,4,1,4,1,4,3,4,
        75,8,4,1,4,0,1,4,5,0,2,4,6,8,0,0,79,0,10,1,0,0,0,2,38,1,0,0,0,4,
        40,1,0,0,0,6,67,1,0,0,0,8,74,1,0,0,0,10,11,6,0,-1,0,11,12,3,2,1,
        0,12,13,6,0,-1,0,13,1,1,0,0,0,14,15,5,13,0,0,15,16,5,1,0,0,16,17,
        3,4,2,0,17,18,6,1,-1,0,18,39,1,0,0,0,19,20,5,2,0,0,20,21,3,8,4,0,
        21,22,5,3,0,0,22,39,1,0,0,0,23,24,5,4,0,0,24,25,3,4,2,0,25,26,5,
        5,0,0,26,27,6,1,-1,0,27,28,3,2,1,0,28,29,6,1,-1,0,29,39,1,0,0,0,
        30,31,5,6,0,0,31,32,6,1,-1,0,32,33,3,4,2,0,33,34,6,1,-1,0,34,35,
        5,7,0,0,35,36,3,2,1,0,36,37,6,1,-1,0,37,39,1,0,0,0,38,14,1,0,0,0,
        38,19,1,0,0,0,38,23,1,0,0,0,38,30,1,0,0,0,39,3,1,0,0,0,40,41,6,2,
        -1,0,41,42,3,6,3,0,42,60,1,0,0,0,43,44,10,4,0,0,44,45,5,8,0,0,45,
        46,3,6,3,0,46,47,6,2,-1,0,47,59,1,0,0,0,48,49,10,3,0,0,49,50,5,9,
        0,0,50,51,3,6,3,0,51,52,6,2,-1,0,52,59,1,0,0,0,53,54,10,2,0,0,54,
        55,5,10,0,0,55,56,3,6,3,0,56,57,6,2,-1,0,57,59,1,0,0,0,58,43,1,0,
        0,0,58,48,1,0,0,0,58,53,1,0,0,0,59,62,1,0,0,0,60,58,1,0,0,0,60,61,
        1,0,0,0,61,5,1,0,0,0,62,60,1,0,0,0,63,64,5,12,0,0,64,68,6,3,-1,0,
        65,66,5,13,0,0,66,68,6,3,-1,0,67,63,1,0,0,0,67,65,1,0,0,0,68,7,1,
        0,0,0,69,70,3,2,1,0,70,71,5,11,0,0,71,72,3,8,4,0,72,75,1,0,0,0,73,
        75,3,2,1,0,74,69,1,0,0,0,74,73,1,0,0,0,75,9,1,0,0,0,5,38,58,60,67,
        74
    ]

class Expr4Parser ( Parser ):

    grammarFileName = "Expr4.g4"

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

        def statement(self):
            return self.getTypedRuleContext(Expr4Parser.StatementContext,0)


        def getRuleIndex(self):
            return Expr4Parser.RULE_start

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

        localctx = Expr4Parser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        try:
            self.enterOuterAlt(localctx, 1)
            self.unique_id=10000
            self.state = 11
            self.statement()

            print("HALT") 
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
            self._Identifier = None # Token

        def Identifier(self):
            return self.getToken(Expr4Parser.Identifier, 0)

        def expr(self):
            return self.getTypedRuleContext(Expr4Parser.ExprContext,0)


        def opt_stmts(self):
            return self.getTypedRuleContext(Expr4Parser.Opt_stmtsContext,0)


        def statement(self):
            return self.getTypedRuleContext(Expr4Parser.StatementContext,0)


        def getRuleIndex(self):
            return Expr4Parser.RULE_statement

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

        localctx = Expr4Parser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 38
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13]:
                self.enterOuterAlt(localctx, 1)
                self.state = 14
                localctx._Identifier = self.match(Expr4Parser.Identifier)
                self.state = 15
                self.match(Expr4Parser.T__0)
                self.state = 16
                self.expr(0)
                print("LVALUE "+(None if localctx._Identifier is None else localctx._Identifier.text)) 
                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 19
                self.match(Expr4Parser.T__1)
                self.state = 20
                self.opt_stmts()
                self.state = 21
                self.match(Expr4Parser.T__2)
                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 3)
                self.state = 23
                self.match(Expr4Parser.T__3)
                self.state = 24
                self.expr(0)
                self.state = 25
                self.match(Expr4Parser.T__4)

                label = str(self.unique_id)
                self.unique_id += 1
                print("GOFALSE "+label)
                          
                self.state = 27
                self.statement()
                print("LABEL "+label)
                          
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 4)
                self.state = 30
                self.match(Expr4Parser.T__5)

                label1 = str(self.unique_id)
                self.unique_id += 1
                label2 = str(self.unique_id)
                self.unique_id += 1
                print("LABEL "+label1)
                                       
                self.state = 32
                self.expr(0)

                print("GOFALSE "+label2)
                                       
                self.state = 34
                self.match(Expr4Parser.T__6)
                self.state = 35
                self.statement()

                print("GOTO "+label1)
                print("LABEL "+label2)
                                       
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

        def term(self):
            return self.getTypedRuleContext(Expr4Parser.TermContext,0)


        def expr(self):
            return self.getTypedRuleContext(Expr4Parser.ExprContext,0)


        def getRuleIndex(self):
            return Expr4Parser.RULE_expr

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
        localctx = Expr4Parser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 41
            self.term()
            self._ctx.stop = self._input.LT(-1)
            self.state = 60
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 58
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
                    if la_ == 1:
                        localctx = Expr4Parser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 43
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 44
                        self.match(Expr4Parser.T__7)
                        self.state = 45
                        self.term()
                        print("IADD") 
                        pass

                    elif la_ == 2:
                        localctx = Expr4Parser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 48
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 49
                        self.match(Expr4Parser.T__8)
                        self.state = 50
                        self.term()
                        print("ISUB") 
                        pass

                    elif la_ == 3:
                        localctx = Expr4Parser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 53
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 54
                        self.match(Expr4Parser.T__9)
                        self.state = 55
                        self.term()
                        print("CMPGT") 
                        pass

             
                self.state = 62
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

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
            self._Number = None # Token
            self._Identifier = None # Token

        def Number(self):
            return self.getToken(Expr4Parser.Number, 0)

        def Identifier(self):
            return self.getToken(Expr4Parser.Identifier, 0)

        def getRuleIndex(self):
            return Expr4Parser.RULE_term

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

        localctx = Expr4Parser.TermContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_term)
        try:
            self.state = 67
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [12]:
                self.enterOuterAlt(localctx, 1)
                self.state = 63
                localctx._Number = self.match(Expr4Parser.Number)
                print("PUSH "+(None if localctx._Number is None else localctx._Number.text)) 
                pass
            elif token in [13]:
                self.enterOuterAlt(localctx, 2)
                self.state = 65
                localctx._Identifier = self.match(Expr4Parser.Identifier)
                print("RVALUE "+(None if localctx._Identifier is None else localctx._Identifier.text)) 
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

        def statement(self):
            return self.getTypedRuleContext(Expr4Parser.StatementContext,0)


        def opt_stmts(self):
            return self.getTypedRuleContext(Expr4Parser.Opt_stmtsContext,0)


        def getRuleIndex(self):
            return Expr4Parser.RULE_opt_stmts

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

        localctx = Expr4Parser.Opt_stmtsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_opt_stmts)
        try:
            self.state = 74
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 69
                self.statement()
                self.state = 70
                self.match(Expr4Parser.T__10)
                self.state = 71
                self.opt_stmts()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 73
                self.statement()
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
                return self.precpred(self._ctx, 4)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 3)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 2)
         




