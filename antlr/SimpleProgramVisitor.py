# Generated from SimpleProgram.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SimpleProgramParser import SimpleProgramParser
else:
    from SimpleProgramParser import SimpleProgramParser

# This class defines a complete generic visitor for a parse tree produced by SimpleProgramParser.

class SimpleProgramVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SimpleProgramParser#start.
    def visitStart(self, ctx:SimpleProgramParser.StartContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#assignmentStatement.
    def visitAssignmentStatement(self, ctx:SimpleProgramParser.AssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#blockStatement.
    def visitBlockStatement(self, ctx:SimpleProgramParser.BlockStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#ifStatement.
    def visitIfStatement(self, ctx:SimpleProgramParser.IfStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#whileStatement.
    def visitWhileStatement(self, ctx:SimpleProgramParser.WhileStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#unaryExpr.
    def visitUnaryExpr(self, ctx:SimpleProgramParser.UnaryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#binaryExpr.
    def visitBinaryExpr(self, ctx:SimpleProgramParser.BinaryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#term.
    def visitTerm(self, ctx:SimpleProgramParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SimpleProgramParser#opt_stmts.
    def visitOpt_stmts(self, ctx:SimpleProgramParser.Opt_stmtsContext):
        return self.visitChildren(ctx)



del SimpleProgramParser