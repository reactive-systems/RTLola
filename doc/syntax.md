# StreamLAB Syntax

This document is a reference for the syntax of StreamLAB specifications.


## Stream Declarations

### Input Streams

`input NAME : TYPE`

```
input a: Bool
input b: String
```

### Output Streams

`output NAME [: TYPE] [@ ACTIVATE] := STREAM_EXPRESSSION`

```
output x: Bool @10Hz := false
output y: Int32 @(a | b) := ...
```

### Activation Conditions

* periodic: `@ FREQUENCY`, e.g., `@1Hz`
	
* variable: `@ BOOLEAN_EXPRESSION` over stream names, e.g., `@ (a & b)`, `@ (a | b)`, ...


## Types

Convention that types are written in CamelCase: `Bool`, `Int64`, `UInt64`, `Float64`, ...


## Expressions

### Literals

`true`, `false`, `NUMERIC`, `"STRING"`

### Unary Operators

Negation `-`, Logical inversion `!`

### Binary Operators

* Artithmetic, e.g., `+`, `-`, `*`
* Boolean, e.g., `&&`, `||`
* Comparision `<`, `=`, `>=` 

### Stream Access

From a stream expression, there are the following ways to refer to a different stream, depending on the *compatibility* of the stream types.

* *direct* access by stream name<br>
  Precondition: Stream types have to be compatible<br>
  Return: The value of the other stream
  
  ```
  input a: Bool
  output x := a
  ```

* *hold* access by using `.hold()`: Returns the last available value<br>
  Precondition: None<br>
  Return: An optional value (there may be no value yet)
  
  ```
  output x @2Hz := ...
  output y @3Hz := x.hold().defaults(to: 0)
  ```

* *optional* access by using `.get()`: Returns the current value if available<br>
  Precondition: Stream types should not be disjunct<br>
  Return: An optional value
  
  ```
  output x @2Hz := ...
  output y @1Hz := x.get().defaults(to: 0)
  ```

### Default

`.defaults(to: VALUE)`

### Offsets

`.offset(by: INTEGER)` and `.offset(by: DURATION)`<br>
alternative: `[INTEGER]` and `[DURATION]`

```
x.offset(by: -1)       // x[-1]
y.offset(by: -100sec)  // y[-100sec]
```

### Sliding Windows

`.aggregate(over: DURATION, using: AGGREGATOR)`

```
x.aggregate(over: 1h, using: sum)
```

possible aggregates are `count`, `sum`, `average`, and `integral`
