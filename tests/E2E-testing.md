# E2E-testing

## Execution

Run the `streamlab-e2e-tests.py` with Python >=3.5 from this folder or the repo base.



## Test definition

You need a specification file and an input file.

Each test case is represented as a file with the `streamlab_test` suffix in this folder.

TODO
JSON format (needs extension for specifying the timestamps)

## Testing using PCAP files

The json format accepts a `input_mode` field which may either be `PCAP` or `CSV`.
If it is set to `PCAP` a pcap file can be supplied and the tool is run in ids mode.

A suitable tool for trace generation: [Colasoft Packet Builder](https://www.colasoft.com/packet_builder/)

*Note:* The windows version of the winpcap library only supports the pcap format. (Not pcap nanoseconds)

## Things to test

- activation condition
    - implicit
        - `add_two_int.streamlab_test`
    - explicit
        - `add_two_int-explicit_ac.streamlab_test`
        - `conjunct_in_ac.streamlab_test`
- access
    - synchronous
        - `add_two_int.streamlab_test` (E➜E))
    - hold
        - `hold_access.streamlab_test` (E➜E)
    - get
        - `get_access.streamlab_test` (E➜E)
    - offset
        - `medium_offset.streamlab_test` (E➜E)
        - `large_offset.streamlab_test` (E➜E)
        - `short_offset-holes.streamlab_test` (E➜E)
- sliding windows
    - aggregations
        - count
            - `simple_sliding_window.streamlab_test` (T➜E)
        - sum
            - `simple_sliding_window.streamlab_test` (T➜E)
        - average
            - `simple_sliding_window.streamlab_test` (T➜E)
        - integral
            - `simple_sliding_window.streamlab_test` (T➜E)
- stream kinds
    - event-based
        - `add_two_int.streamlab_test`
        - ...
    - frequency-based
        - `frequencies.streamlab_test` (T➜E) (T➜T)

## Things to do

- add time based checks for triggers ( time since start / absolute time)
- add tolerance for time based trigger checking
- adjust `simple_sliding_window.streamlab_test` for the other aggregations
- tuple, and data types, string, regexp
- forward sliding windows / delay operator
- min/max
- let

agg(over:9s) agg(over: =9s)
