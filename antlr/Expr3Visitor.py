# Generated from Expr3.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr3Parser import Expr3Parser
else:
    from Expr3Parser import Expr3Parser

# This class defines a complete generic visitor for a parse tree produced by Expr3Parser.

class Expr3Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by Expr3Parser#expr.
    def visitExpr(self, ctx:Expr3Parser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr3Parser#term.
    def visitTerm(self, ctx:Expr3Parser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr3Parser#number.
    def visitNumber(self, ctx:Expr3Parser.NumberContext):
        return self.visitChildren(ctx)



del Expr3Parser