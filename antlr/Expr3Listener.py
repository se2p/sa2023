# Generated from Expr3.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr3Parser import Expr3Parser
else:
    from Expr3Parser import Expr3Parser

# This class defines a complete listener for a parse tree produced by Expr3Parser.
class Expr3Listener(ParseTreeListener):

    # Enter a parse tree produced by Expr3Parser#expr.
    def enterExpr(self, ctx:Expr3Parser.ExprContext):
        pass

    # Exit a parse tree produced by Expr3Parser#expr.
    def exitExpr(self, ctx:Expr3Parser.ExprContext):
        pass


    # Enter a parse tree produced by Expr3Parser#term.
    def enterTerm(self, ctx:Expr3Parser.TermContext):
        pass

    # Exit a parse tree produced by Expr3Parser#term.
    def exitTerm(self, ctx:Expr3Parser.TermContext):
        pass


    # Enter a parse tree produced by Expr3Parser#number.
    def enterNumber(self, ctx:Expr3Parser.NumberContext):
        pass

    # Exit a parse tree produced by Expr3Parser#number.
    def exitNumber(self, ctx:Expr3Parser.NumberContext):
        pass



del Expr3Parser