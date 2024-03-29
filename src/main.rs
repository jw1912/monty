use monty::{ataxx, UciLike};

#[cfg(not(feature = "ataxx"))]
use monty::{chess, TunableParams};

fn main() {
    #[cfg(not(feature = "ataxx"))]
    {
        let mut args = std::env::args();
        let params = TunableParams::default();

        #[cfg(not(feature = "ataxx"))]
        match args.nth(1).as_deref() {
            Some("bench") => chess::Uci::bench(5, &chess::POLICY_NETWORK, &chess::NNUE, &params),
            Some("ataxx") => ataxx::Uai::run(&(), &()),
            Some(_) => println!("unknown mode!"),
            None => chess::Uci::run(&chess::POLICY_NETWORK, &chess::NNUE),
        }
    }


    #[cfg(feature = "ataxx")]
    ataxx::Uai::run(&(), &())
}
