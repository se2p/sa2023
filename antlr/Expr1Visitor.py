# Generated from Expr1.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr1Parser import Expr1Parser
else:
    from Expr1Parser import Expr1Parser

# This class defines a complete generic visitor for a parse tree produced by Expr1Parser.

class Expr1Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by Expr1Parser#expr.
    def visitExpr(self, ctx:Expr1Parser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr1Parser#term.
    def visitTerm(self, ctx:Expr1Parser.TermContext):
        return self.visitChildren(ctx)



del Expr1Parser