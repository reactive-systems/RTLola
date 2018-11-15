# Types

This document gives an overview over the type system used in Lola.
As Lola is a data manipulation language, we do not have references/pointers and everything is call-by-value.

## Primitive Types 

The easiest class of types, typically with direct correspondence to machine instructions.

All type names are supposed to be capitalized.

* Numeric types:
  * the name consists of the type itself, i.e. `Float` for floating point
  numbers, `Int` for signed integers, and `UInt` for unsigned integers, followed
  by the bit-width
  * integers can have a width of `8`, `16`, `32`, or `64`
  * floating point numbers can have a width of either `32` or `64`
  * examples: `Float32`, `UInt8`, `Int16`
  * overflow operations and arithmetically invalid operations such as division by `0` have no specified numeric result
  * there is a guarantee that no arithmetic operation will result in a panic/crash
  * an implicit `Bool`-typed stream with name `error` indicates that an operation without specified result was taken out:
  ```
  input in: Int8
  output out: Int8 := in/(out[-1] ? 0)
  trigger error
  //      ^^^^^
  //         will trigger in the first evaluation cycle
  ```
  * types will implicitly be widened, e.g., `Int16` will implicitly be converted into `Int32` and `Int64` if necessary
  * there is no other implicit conversion, this especially includes that specifications such as the following are invalid:
  ```
  input uint_input: UInt8
  output int_output: Int8 := uint_input
  //                         ^^^^^^^^^^
  //                              error occurs here; incompatible types
  ```
  * explicit conversions are possible by explicit constructor calls: 
  ```
  input uint_input: UInt8
  output int_output: Int8 := Int8(uint_input) // works just fine
  ```
  * invalid conversions at runtime are reported on the error stream, the result is unspecified

* Logic type:
  * the type `Bool` can have the value `true` or `false`

* Strings:
  * type `String` represents a zero-terminated UTF-8 encoded array of bytes.
  * there is no support of a character-based type

* Decimals:
  * probably in a future release


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