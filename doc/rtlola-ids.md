# Intrusion Detection System

## Usage
To use `rtlola-interpreter` as an intrusion detection system use the subcommand `ids`. It requires you to input the lola specefication file, a description the local network space and either an pcap file for offline monitoring a network interface to listen on. For more information refer to the help page.

## Available Input Streams and their Types

The input streams need to be declared as usual in the specification file. Their names are composed as follows: `<Protocol>::<Header Field>`. For example:

`input Ethernet::source: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)`

Will provide the input stream with the current source mac addresses.

Header flags can be accessed as follows: `<Protocol>::flags::<Flag name>`. For example:

`input IPv4::flags::df: Bool`

### Ethernet

```
destination: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
source: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
etype: UInt16
```

### IPv4

```
source: (UInt8, UInt8, UInt8, UInt8)
destination: (UInt8, UInt8, UInt8, UInt8)
ihl: UInt8
dscp: UInt8
ecn: UInt8
length: UInt16
identification: UInt16
fragment_offset: UInt16
ttl: UInt8
protocol: UInt8
checksum: UInt16
```
IPv4 flags:
```
df: Bool
mf: Bool
```

### IPv6

```
source: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, 
         UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
destination: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, 
              UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
traffic_class: UInt8
flow_label: UInt32
length: UInt16
hop_limit: UInt8
```

### TCP

```
source: UInt16
destination: UInt16
seq_number: UInt32
ack_number: UInt32
data_offset: UInt8
window_size: UInt16
checksum: UInt16
urgent_pointer: UInt16
```

TCP flags

```
ns: bool
cwr: bool
ece: bool
urg: bool
ack: bool
psh: bool
rst: bool
syn: bool
fin: bool
```

### UDP

```
source: UInt16
destination: UInt16
length: UInt16
checksum: UInt16
```

### Auxiliary Input Streams

```
payload: String
```
The payload stream contains the payload of each packet as an UTF-8 encoded string.
```
direction: String
```
The direction stream can either take the value `Incoming` or `Outgoing`, depending on wether the destination ip of a packet belongs to the given local network.
```
protocol: String
```
The protocol stream contains the name of the highest level recognized protocol. The possible values are:
- TCP
- UDP
- IPv4
- IPv6
- Ethernet2
- Unknown