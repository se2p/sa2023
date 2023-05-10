grammar SimpleProgram;

start : statement
      ;

statement : Identifier ':=' expr        # assignmentStatement
          | 'begin' opt_stmts 'end'     # blockStatement
          | 'if' expr 'then' statement  # ifStatement
          | 'while' expr 'do' statement # whileStatement
          ;

expr : expr op=('+' | '-' | '>') term  # binaryExpr
     | term                      # unaryExpr
     ;

term : Number
     | Identifier
     ;

opt_stmts : statement ';' opt_stmts
          | statement
          ;

Number : Digit+
       ;

Identifier : [a-zA-Z_] [a-zA-Z_0-9]*
           ;

Digit : ('0'..'9') ;
WS : [ \t\r\n]+ -> skip ;
