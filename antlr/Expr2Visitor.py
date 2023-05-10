# Generated from Expr2.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr2Parser import Expr2Parser
else:
    from Expr2Parser import Expr2Parser

# This class defines a complete generic visitor for a parse tree produced by Expr2Parser.

class Expr2Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by Expr2Parser#expr.
    def visitExpr(self, ctx:Expr2Parser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr2Parser#term.
    def visitTerm(self, ctx:Expr2Parser.TermContext):
        return self.visitChildren(ctx)



del Expr2Parser