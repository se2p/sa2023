grammar Expr2;

expr : expr '+' term {print("+", end='')} |
       expr '-' term {print("-", end='')} |
       term;

term : DIGIT {print($DIGIT.text, end='') } ;

DIGIT : ('0'..'9') ;
WS : [ \t\r\n]+ -> skip ;

