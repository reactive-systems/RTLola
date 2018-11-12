extern crate proc_macro;

#[macro_use]
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;

use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
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
                & self._span
            }
        }
    })
}

#[proc_macro_attribute]
pub fn ast_node(_metadata: TokenStream, input: TokenStream) -> TokenStream {
    let mut input: syn::ItemStruct = parse_macro_input!(input as syn::ItemStruct);

    let copy_source: syn::ItemStruct = parse_quote!{
        struct source {
            pub(crate) _id: NodeId,
            pub _span: Span,
        }
    };
    if let syn::Fields::Named(new_fields) = copy_source.fields {
        input.attrs.push(parse_quote!{#[derive(AstNode, Debug)]});
        match input.fields {
            syn::Fields::Named(ref mut named_fields) => {
                for new_field in new_fields.named.iter() {
                    named_fields.named.push(new_field.clone());
                }
            }
            syn::Fields::Unnamed(_) | syn::Fields::Unit => {
                //                input.span().
                //                    .unstable()
                //                    .error("Only allowed on structs with named fields.")
                //                    .emit()
            }
        }
    } else {
        unreachable!()
    }

    TokenStream::from(quote!(#input))
}
