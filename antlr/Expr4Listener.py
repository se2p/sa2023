# Generated from Expr4.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr4Parser import Expr4Parser
else:
    from Expr4Parser import Expr4Parser

# This class defines a complete listener for a parse tree produced by Expr4Parser.
class Expr4Listener(ParseTreeListener):

    # Enter a parse tree produced by Expr4Parser#start.
    def enterStart(self, ctx:Expr4Parser.StartContext):
        pass

    # Exit a parse tree produced by Expr4Parser#start.
    def exitStart(self, ctx:Expr4Parser.StartContext):
        pass


    # Enter a parse tree produced by Expr4Parser#statement.
    def enterStatement(self, ctx:Expr4Parser.StatementContext):
        pass

    # Exit a parse tree produced by Expr4Parser#statement.
    def exitStatement(self, ctx:Expr4Parser.StatementContext):
        pass


    # Enter a parse tree produced by Expr4Parser#expr.
    def enterExpr(self, ctx:Expr4Parser.ExprContext):
        pass

    # Exit a parse tree produced by Expr4Parser#expr.
    def exitExpr(self, ctx:Expr4Parser.ExprContext):
        pass


    # Enter a parse tree produced by Expr4Parser#term.
    def enterTerm(self, ctx:Expr4Parser.TermContext):
        pass

    # Exit a parse tree produced by Expr4Parser#term.
    def exitTerm(self, ctx:Expr4Parser.TermContext):
        pass


    # Enter a parse tree produced by Expr4Parser#opt_stmts.
    def enterOpt_stmts(self, ctx:Expr4Parser.Opt_stmtsContext):
        pass

    # Exit a parse tree produced by Expr4Parser#opt_stmts.
    def exitOpt_stmts(self, ctx:Expr4Parser.Opt_stmtsContext):
        pass



del Expr4Parser