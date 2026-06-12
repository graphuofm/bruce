//! Bruce CLI — `bruce` command-line tool.
//!
//! ```bash
//! bruce demo                  # run a 3-record attention demo
//! bruce version               # print version
//! ```

use anyhow::Result;
use bruce_core::{Eps, F_eps, Sim};
use clap::{Parser, Subcommand};
use ndarray::array;

#[derive(Parser, Debug)]
#[command(
    name = "bruce",
    about = "Bruce: a unified algebra of relational databases and attention",
    version,
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run a small attention demo to verify the build works.
    Demo,
    /// Print the bruce-core version.
    Version,
}

fn cmd_demo() -> Result<()> {
    println!("Bruce CLI demo — F_ε attention on a 3-record memory");
    let x = array![1.0, 0.0];
    let k = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let v = array![[10.0, 0.0], [0.0, 20.0], [5.0, 5.0]];

    for (label, eps) in [
        ("ε = 0  (tropical / SQL)", Eps::ZERO),
        ("ε = 0.25", Eps::QUARTER),
        ("ε = 1.0  (softmax)", Eps::ONE),
        ("ε = 4.0", Eps(4.0)),
    ] {
        let sim = if eps.is_zero() { Sim::Indicator } else { Sim::Dot };
        let op = F_eps::new(eps, sim);
        let out = op.attention(&x.view(), &k.view(), &v.view());
        println!("  {label}: out = {:?}", out.as_slice().unwrap());
    }

    println!("\nAll done.");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Demo => cmd_demo()?,
        Cmd::Version => {
            println!("bruce-cli {}", env!("CARGO_PKG_VERSION"));
            println!("bruce-core 0.1.0");
        }
    }
    Ok(())
}
