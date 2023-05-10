grammar Expr1;

expr : expr '+' term  |
       expr '-' term  |
       term;

term : DIGIT ;

DIGIT : ('0'..'9') ;
WS : [ \t\r\n]+ -> skip ;

