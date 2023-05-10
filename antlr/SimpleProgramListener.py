# Generated from SimpleProgram.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SimpleProgramParser import SimpleProgramParser
else:
    from SimpleProgramParser import SimpleProgramParser

# This class defines a complete listener for a parse tree produced by SimpleProgramParser.
class SimpleProgramListener(ParseTreeListener):

    # Enter a parse tree produced by SimpleProgramParser#start.
    def enterStart(self, ctx:SimpleProgramParser.StartContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#start.
    def exitStart(self, ctx:SimpleProgramParser.StartContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#assignmentStatement.
    def enterAssignmentStatement(self, ctx:SimpleProgramParser.AssignmentStatementContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#assignmentStatement.
    def exitAssignmentStatement(self, ctx:SimpleProgramParser.AssignmentStatementContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#blockStatement.
    def enterBlockStatement(self, ctx:SimpleProgramParser.BlockStatementContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#blockStatement.
    def exitBlockStatement(self, ctx:SimpleProgramParser.BlockStatementContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#ifStatement.
    def enterIfStatement(self, ctx:SimpleProgramParser.IfStatementContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#ifStatement.
    def exitIfStatement(self, ctx:SimpleProgramParser.IfStatementContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#whileStatement.
    def enterWhileStatement(self, ctx:SimpleProgramParser.WhileStatementContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#whileStatement.
    def exitWhileStatement(self, ctx:SimpleProgramParser.WhileStatementContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#unaryExpr.
    def enterUnaryExpr(self, ctx:SimpleProgramParser.UnaryExprContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#unaryExpr.
    def exitUnaryExpr(self, ctx:SimpleProgramParser.UnaryExprContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#binaryExpr.
    def enterBinaryExpr(self, ctx:SimpleProgramParser.BinaryExprContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#binaryExpr.
    def exitBinaryExpr(self, ctx:SimpleProgramParser.BinaryExprContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#term.
    def enterTerm(self, ctx:SimpleProgramParser.TermContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#term.
    def exitTerm(self, ctx:SimpleProgramParser.TermContext):
        pass


    # Enter a parse tree produced by SimpleProgramParser#opt_stmts.
    def enterOpt_stmts(self, ctx:SimpleProgramParser.Opt_stmtsContext):
        pass

    # Exit a parse tree produced by SimpleProgramParser#opt_stmts.
    def exitOpt_stmts(self, ctx:SimpleProgramParser.Opt_stmtsContext):
        pass



del SimpleProgramParser