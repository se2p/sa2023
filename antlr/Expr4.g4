grammar Expr4;

start : {self.unique_id=10000} statement {
print("HALT") }
      ;

statement : Identifier ':=' expr  {print("LVALUE "+$Identifier.text) }
          | 'begin' opt_stmts 'end'
          | 'if' expr 'then' {
label = str(self.unique_id)
self.unique_id += 1
print("GOFALSE "+label)
          } statement {print("LABEL "+label)
          }
          | 'while' {
label1 = str(self.unique_id)
self.unique_id += 1
label2 = str(self.unique_id)
self.unique_id += 1
print("LABEL "+label1)
                       }
                       expr {
print("GOFALSE "+label2)
                       }
                      'do' statement {
print("GOTO "+label1)
print("LABEL "+label2)
                       }
          ;

expr : expr '+' term {print("IADD") }
     | expr '-' term {print("ISUB") }
     | expr '>' term  {print("CMPGT") }
     | term
     ;

term : Number  {print("PUSH "+$Number.text) }
     | Identifier  {print("RVALUE "+$Identifier.text) }
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
