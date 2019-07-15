use std::fs;
use std::path::PathBuf;

use crate::analysis;
use crate::parse::SourceMapper;
use crate::reporting::Handler;
use crate::ty::TypeConfig;

pub fn analyze(filename: &str) {
    let contents = fs::read_to_string(filename).unwrap_or_else(|e| {
        eprintln!("Could not read file `{}`: {}", filename, e);
        std::process::exit(1)
    });
    let mapper = SourceMapper::new(PathBuf::from(filename), &contents);
    let handler = Handler::new(mapper);
    let spec = crate::parse::parse(&contents, &handler).unwrap_or_else(|e| {
        eprintln!("parse error:\n{}", e);
        std::process::exit(1)
    });
    let _report = analysis::analyze(&spec, &handler, TypeConfig { use_64bit_only: false, type_aliases: false });
    //println!("{:?}", report);
    //use crate::analysis::graph_based_analysis::MemoryBound;
    //report.graph_analysis_result.map(|r| match r.memory_requirements {
    //    MemoryBound::Unbounded => println!("The specification has no bound on the memory consumption."),
    //    MemoryBound::Bounded(bytes) => println!("The specification uses at most {} bytes.", bytes),
    //    MemoryBound::Unknown => println!("Incomplete specification: we cannot determine the memory consumption."),
    //});
}
