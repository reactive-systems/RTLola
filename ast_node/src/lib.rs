#[macro_use]
extern crate pest;
#[macro_use]
extern crate pest_derive;
//use pest;

/// Every node in the AST gets a unique id, represented by a 32bit unsiged integer.
/// They are used in the later analysis phases to store information about AST nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub fn new(x: usize) -> NodeId {
        assert!(x < (std::u32::MAX as usize));
        NodeId(x as u32)
    }

    pub fn from_u32(x: u32) -> NodeId {
        NodeId(x)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// When parsing, we initially give all AST nodes this AST node id.
    /// Then later, in the renumber pass, we renumber them to have small, positive ids.
    pub const DUMMY: NodeId = NodeId(!0);
}

/// A span marks a range in a file.
/// Start and end positions are *byte* offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    // TODO Do we need this here or do we want to keep a mapping from byte positions to lines in the LSP part.
    // line: usize,
    // /// The LSP uses UTF-16 code units (2 bytes) as their unit for offsets.
    // lineOffsetLSP: usize,
}

impl<'a> From<pest::Span<'a>> for Span {
    fn from(span: pest::Span<'a>) -> Self {
        Span {
            start: span.start(),
            end: span.end(),
        }
    }
}

pub trait AstNode<'a> {
    fn id(&'a self) -> &'a NodeId;
    fn span(&'a self) -> &'a Span;
}
