grammar SimpleProgramAttributed;

start returns [ASTNode node]
      : statement {$node = $statement.node }
      ;

statement returns [ASTNode node]
          : Identifier ':=' expr        {$node = AssignmentStatement(Identifier($Identifier.text), $expr.node) }
          | 'begin' opt_stmts 'end'     {$node = BlockStatement($opt_stmts.nodes) }
          | 'if' a=expr 'then' statement  {$node = IfStatement($a.node, $statement.node) }
          | 'while' a=expr 'do' statement {$node = WhileStatement($a.node, $statement.node) }
          ;

expr returns [ASTNode node]
     : a=expr op=('+' | '-' | '>') term  {$node = Expression($a.node, $term.node, $op.text) }
     | term                            {$node = $term.node }
     ;

term returns [ASTNode node]
     : Number      {$node = Number($Number.text) }
     | Identifier  {$node = Identifier($Identifier.text) }
     ;

opt_stmts returns [java.util.List<ASTNode> nodes]
          : statement ';' opt_stmts  {$nodes = [ $statement.node] + $opt_stmts.nodes }
          | statement                {$nodes = [ $statement.node] }
          ;

Number : Digit+
       ;

Identifier : [a-zA-Z_] [a-zA-Z_0-9]*
           ;

Digit : ('0'..'9') ;
WS : [ \t\r\n]+ -> skip ;
