# Generated from SimpleProgramAttributed.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SimpleProgramAttributedParser import SimpleProgramAttributedParser
else:
    from SimpleProgramAttributedParser import SimpleProgramAttributedParser

# This class defines a complete generic visitor for a parse tree produced by SimpleProgramAttributedParser.

class SimpleProgramAttributedVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SimpleProgramAttributedParser#start.
    def visitStart(self, ctx:SimpleProgramAttributedParser.StartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramAttributedParser#statement.
    def visitStatement(self, ctx:SimpleProgramAttributedParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramAttributedParser#expr.
    def visitExpr(self, ctx:SimpleProgramAttributedParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramAttributedParser#term.
    def visitTerm(self, ctx:SimpleProgramAttributedParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramAttributedParser#opt_stmts.
    def visitOpt_stmts(self, ctx:SimpleProgramAttributedParser.Opt_stmtsContext):
        return self.visitChildren(ctx)



del SimpleProgramAttributedParser