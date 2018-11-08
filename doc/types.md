# Types

This document gives an overview over the type system used in Lola.
As Lola is a data manipulation language, we do not have references/pointers and everything is call-by-value.

## Primitive Types (WIP)

The easiest class of types, typically with direct correspondence to machine instructions.

* integer:
  * bit-width 8, 16, 32, and 64
  * signed and unsigned
  
  Questions:
  * how to name them? C like int, uint, int, int16_t, Swift like Int, UInt, Int16, or Rust like i32, u32, other?
  * overflow semantic?
  * implicit/explicit conversion? Rust for example has no undefined behavior when converting primitive types:<https://doc.rust-lang.org/stable/rust-by-example/types/cast.html?highlight=Overf#casting>
  * other questions?

* floating point?
* decimal?
* string and char?

## Product Types (future work)

Consist of other types, usually identified by a label.
Example: Point consist of x and y coordinate.

Questions:
How to name them? Struct, record, ...?
How to define them?
How to use them as an input?
What to do with them? Operator overloading, methods?
How to access fields? `a.field`

## Sum Types (future work)

Consist of a fixed set of different values.
Example: HTTP request method can be GET, POST, ...

Questions?
How to name them? Enum, datatype, ...?
How to define them?
How to use them as an input?
What to do with them? Operator overloading, methods?
How to use them? Destructing, pattern matching, etc.