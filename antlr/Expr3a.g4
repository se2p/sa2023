grammar Expr3;

start : statement
      ;

statement : Identifier ':=' expr
          | 'begin' opt_stmts 'end'
          | 'if' expr 'then' statement
          | 'while' expr 'do' statement
          ;

expr : expr '+' term
     | expr '-' term
     | term
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
