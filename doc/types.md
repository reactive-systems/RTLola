# Types

This document gives an overview over the type system used in Lola.
As Lola is a data manipulation language, we do not have references/pointers and everything is call-by-value.

> Notation: Types are written in CamelCase

## Primitive Types 

### Integers

We distinguish between signed and unsigned integer types, using the `Int` and `UInt` prefix, followed by the bit-width, e.g. `UInt16`.

Notes:

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

### Floating Point

There are two floating point types, `Float32` and `Float64`.

### Boolean

The boolean type is `Bool` and has values `true` and `false`.

### Strings

The type `String` represents a zero-terminated UTF-8 encoded array of bytes.
There is no support of a character-based type.

### Tuples

Tuple types are represented by `(Int32, Bool)`, their values can be constructed by using `(1, true)`. 
Tuples can be accessed by index `t.0`, where the first index is `0`.

## Stream Types

### Event Streams

An event triggered stream has the type `EventStream<T, (T1,T2,..,Tn)>` where `T` is the type of the value stored in the stream and `(T1,T2,..,Tn)` is the type of the parameterization.

## Abstract Data Types (future work)



## Typing Rules

* Input: $\dfrac{\textbf{input}~\textit{in} : T}{\textit{in} : \text{EventStream<T>}}$