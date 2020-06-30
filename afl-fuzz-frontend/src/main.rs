#[macro_use]
extern crate afl;
extern crate rtlola_frontend;
use rtlola_frontend::FrontendConfig;

fn main() {
    fuzz!(|data: &[u8]| {
        if let Ok(s) = std::str::from_utf8(data) {
            let _ = rtlola_frontend::parse("stdin", s, FrontendConfig::default());
        }
    });
}
