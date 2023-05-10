grammar Expr3;

expr : expr '+' term {print("+ ", end='')} |
       expr '-' term {print("- ", end='')} |
       term;

term : number  {print($number.text, "", end='') } ;

number: DIGIT+;

DIGIT : ('0'..'9') ;
WS : [ \t\r\n]+ -> skip ;
