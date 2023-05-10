# Generated from Expr4.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .Expr4Parser import Expr4Parser
else:
    from Expr4Parser import Expr4Parser

# This class defines a complete generic visitor for a parse tree produced by Expr4Parser.

class Expr4Visitor(ParseTreeVisitor):

    # Visit a parse tree produced by Expr4Parser#start.
    def visitStart(self, ctx:Expr4Parser.StartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr4Parser#statement.
    def visitStatement(self, ctx:Expr4Parser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr4Parser#expr.
    def visitExpr(self, ctx:Expr4Parser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr4Parser#term.
    def visitTerm(self, ctx:Expr4Parser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by Expr4Parser#opt_stmts.
    def visitOpt_stmts(self, ctx:Expr4Parser.Opt_stmtsContext):
        return self.visitChildren(ctx)



del Expr4Parser