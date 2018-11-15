extern crate proc_macro;

//#[macro_use]
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(AstNode)]
pub fn ast_node_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree.
    let ast = parse_macro_input!(input as DeriveInput);

    // Build the impl
    let gen = impl_ast_node(&ast);

    // Return the generated impl
    gen
    //    gen.parse().unwrap()
}

fn impl_ast_node(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    proc_macro::TokenStream::from(quote! {
        impl<'a> ast_node::AstNode<'a> for #name{
            fn id(&'a self) -> &'a NodeId {
                &self._id
            }

            fn span(&'a self) -> &'a Span {
                &self._span
            }

            fn set_id(&'a mut self, id: NodeId){
                self._id = id
            }
        }
    })
}
