use crate::{
    search::{mcts::Searcher, params::TunableParams, policy::{PolicyNetwork, get_policy}},
    state::position::{self, Position},
};

use std::time::Instant;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

pub fn preamble() {
    println!("id name monty {}", env!("CARGO_PKG_VERSION"));
    println!("id author Jamie Whiting");
    println!("option name report_moves type button");
    TunableParams::uci_info();
    println!("uciok");
}

pub fn isready() {
    println!("readyok");
}

pub fn setoption(commands: &[&str], params: &mut TunableParams, report_moves: &mut bool) {
    if let ["setoption", "name", "report_moves"] = commands {
        *report_moves = !*report_moves;
        return;
    }

    let (name, val) = if let ["setoption", "name", x, "value", y] = commands {
        (*x, y.parse::<i32>().unwrap())
    } else {
        return;
    };

    params.set(name, f64::from(val) / 100.0);
}

pub fn position(commands: Vec<&str>, pos: &mut Position, stack: &mut Vec<u64>) {
    let mut fen = String::new();
    let mut move_list = Vec::new();
    let mut moves = false;

    for cmd in commands {
        match cmd {
            "position" | "fen" => {}
            "startpos" => fen = STARTPOS.to_string(),
            "kiwipete" => fen = KIWIPETE.to_string(),
            "moves" => moves = true,
            _ => {
                if moves {
                    move_list.push(cmd);
                } else {
                    fen.push_str(&format!("{cmd} "));
                }
            }
        }
    }

    *pos = Position::parse_fen(&fen);
    stack.clear();

    for m in move_list {
        stack.push(pos.hash());
        let possible_moves = pos.gen();

        for mov in possible_moves.iter() {
            if m == mov.to_uci() {
                pos.make(*mov);
            }
        }
    }
}

pub fn go(
    commands: &[&str],
    stack: Vec<u64>,
    pos: &Position,
    params: &TunableParams,
    report_moves: bool,
    policy: &PolicyNetwork,
) {
    let mut nodes = 10_000_000;
    let mut max_time = None;
    let mut max_depth = 256;

    let mut mode = "";

    for cmd in commands {
        match *cmd {
            "nodes" => mode = "nodes",
            "movetime" => mode = "movetime",
            "depth" => mode = "depth",
            _ => match mode {
                "nodes" => nodes = cmd.parse().unwrap_or(nodes),
                "movetime" => max_time = cmd.parse().ok(),
                "depth" => max_depth = cmd.parse().unwrap_or(max_depth),
                _ => {}
            },
        }
    }

    let mut searcher = Searcher::new(*pos, stack, nodes, params.clone(), policy);

    let (mov, _) = searcher.search(max_time, max_depth, report_moves, true, &mut 0);

    println!("bestmove {}", mov.to_uci());
}

pub fn eval(pos: &Position, params: &TunableParams, policy: &PolicyNetwork) {
    let moves = pos.gen();
    let mut policies = Vec::new();
    let mut total = 0.0;

    for mov in moves.iter() {
        let pol = get_policy(mov, pos, policy).exp();
        total += pol;
        policies.push(pol);
    }

    for (mov, policy) in moves.iter().zip(policies) {
        println!("{} -> {: >5.2}%", mov.to_uci(), policy / total * 100.0);
    }

    println!(
        "info eval cp {} wdl {:.2}",
        pos.eval_cp(),
        pos.eval(params) * 100.0
    );
}

pub fn perft(commands: &[&str], pos: &Position) {
    let depth = commands[1].parse().unwrap();
    let now = Instant::now();
    let count = position::perft::<false, true>(pos, depth);
    let time = now.elapsed().as_micros();
    println!(
        "perft {depth} time {} nodes {count} ({:.2} Mnps)",
        time / 1000,
        count as f64 / time as f64
    );
}
