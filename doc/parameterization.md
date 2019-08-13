# Parameterization

## Motivation

Parameterization is the ability to group similar data together, to compute properties, and to quantify over them.
For example, in network monitoring, we often compute properties per source IP address.

## Desgin

Readability and familarity is a major factor in the design of parameterization.

### Syntax

Declaration of parametric outputs is given by the grammar

```
OUTPUT      := output NAME ( PARAMETERS ) FILTER TERMINATION : TYPE AC := EXPRESSION
PARAMETERS  := (NAME (: TYPE)?)+
FILTER      := filter EXPRESSION
TERMINATION := close EXPRESSION
```

### Examples

* parametric declaration `output x(a: Int8, b: Bool) : Int8 := 1`
* Parametric outputs can be used as functions, e.g., `output y := x(1, false)`
* filter conditions `output x filter (x != y) := 1`
* termination condition `output x close x > 10 := x[-1].defaults(to: 0)`





 