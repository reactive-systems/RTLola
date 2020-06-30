# RTLola

RTLola is a monitoring framework for reactive systems.

## Installation Notes

If you want to use the network interface, the provided binaries require a PCAP library to be installed. If it is not already installed on your system you can do so as follows:

### Windows

You can download and install the library from here:
[NPcap](https://nmap.org/npcap/)

### Linux

Use the packet manager of your choice to install the `libpcap-dev` package. For example using `apt`:

`apt install libpcap-dev`

### Mac OS

The PCAP library is already be included in Mac OS X.

## Command Line Usage

### Specification Analysis

```
rtlola-interpreter analyze [SPEC]
```

checks whether the given specification is valid

### Monitoring

```
rtlola-interpreter monitor [SPEC] --offline --csv-in [TRACE] --verbosity progress
```

For example, given the specification

```
input a: Int64
input b: Int64

output x := a + b
trigger x > 2
```

in file `example.spec` and the CSV

```
a,b,time
0,1,0.1
2,3,0.2
4,5,0.3
```

in file `example.csv` we get

```
rtlola-interpreter monitor example.spec --offline --csv-in example.csv 
Trigger: x > 2
Trigger: x > 2
```


See all available options with `rtlola-interpreter --help`